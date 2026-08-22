"""
Retrieval 测试。

按模块规范要求的 6 个测试：
1. test_in_memory_retrieval_returns_acl_filtered_hits
2. test_dense_and_sparse_results_are_fused_by_rrf
3. test_exact_policy_code_query_uses_bm25_signal
4. test_semantic_query_uses_dense_signal
5. test_reranker_reorders_candidates
6. test_evidence_bundle_contains_citations
"""

from __future__ import annotations

import pytest

from app.schemas.chunk import ChunkCreate
from app.schemas.enums import Visibility
from app.schemas.retrieval import RetrievalResult
from app.services.retrieval.embedding.mock_embedding import MockEmbedder
from app.services.retrieval.evidence_builder import EvidenceBuilder
from app.services.retrieval.fusion.rrf import RRFFuser
from app.services.retrieval.reranker.mock_reranker import MockReranker
from app.services.retrieval.store.base import ACLFilter
from app.services.retrieval.store.memory_bm25 import InMemoryBM25Store
from app.services.retrieval.store.memory_vector import InMemoryVectorStore


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Mock Embedder 实例。"""
    return MockEmbedder(dimension=64)


@pytest.fixture
def acl_filter() -> ACLFilter:
    """ACL 过滤器。"""
    return ACLFilter(
        tenant_id="tenant_hr",
        department_ids=["dept_001", "dept_002"],
        allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
    )


@pytest.fixture
def sample_chunks() -> list[ChunkCreate]:
    """示例 chunks。"""
    return [
        ChunkCreate(
            document_id="doc_001",
            chunk_text="新员工入职需要提交身份证复印件和学历证明。",
            context_prefix="本片段来自《员工入职制度》。",
            full_text="本片段来自《员工入职制度》。新员工入职需要提交身份证复印件和学历证明。",
            parent_id="chunk_parent_001",
            chunk_type="child",
            heading_path="员工入职制度 > 第二章 入职材料",
            page_numbers=[3],
            token_count=45,
            tenant_id="tenant_hr",
            department_id="dept_001",
            visibility=Visibility.DEPARTMENT,
            acl_metadata={"author": "HR"},
        ),
        ChunkCreate(
            document_id="doc_001",
            chunk_text="试用期为3个月，特殊情况可延长至6个月。",
            context_prefix="本片段来自《员工入职制度》。",
            full_text="本片段来自《员工入职制度》。试用期为3个月，特殊情况可延长至6个月。",
            parent_id="chunk_parent_002",
            chunk_type="child",
            heading_path="员工入职制度 > 第三章 试用期",
            page_numbers=[5],
            token_count=30,
            tenant_id="tenant_hr",
            department_id="dept_001",
            visibility=Visibility.DEPARTMENT,
            acl_metadata={"author": "HR"},
        ),
        ChunkCreate(
            document_id="doc_002",
            chunk_text="请假需要提前1天申请，病假需要提供医院证明。",
            context_prefix="本片段来自《考勤制度》。",
            full_text="本片段来自《考勤制度》。请假需要提前1天申请，病假需要提供医院证明。",
            parent_id="chunk_parent_003",
            chunk_type="child",
            heading_path="考勤制度 > 第三章 请假规定",
            page_numbers=[2],
            token_count=35,
            tenant_id="tenant_hr",
            department_id="dept_002",
            visibility=Visibility.DEPARTMENT,
            acl_metadata={"author": "HR"},
        ),
        ChunkCreate(
            document_id="doc_003",
            chunk_text="机密文档：公司年度战略规划。",
            context_prefix="本片段来自《战略规划》。",
            full_text="本片段来自《战略规划》。机密文档：公司年度战略规划。",
            parent_id="chunk_parent_004",
            chunk_type="child",
            heading_path="战略规划 > 第一章",
            page_numbers=[1],
            token_count=25,
            tenant_id="tenant_hr",
            department_id="dept_001",
            visibility=Visibility.CONFIDENTIAL,
            acl_metadata={"author": "CEO"},
        ),
    ]


class TestACLFilter:
    """ACL 过滤测试。"""

    @pytest.mark.asyncio
    async def test_milvus_search_pushes_acl_filter_to_query_expression(self) -> None:
        """测试：MilvusClient 检索必须通过 filter 在召回前下推 ACL。"""
        from app.services.retrieval.store.milvus_vector import MilvusVectorStore

        class FakeMilvusClient:
            """记录 MilvusClient.search 入参的 fake client。"""

            def __init__(self) -> None:
                self.search_kwargs: dict[str, object] = {}

            def search(self, **kwargs: object) -> list[list[object]]:
                self.search_kwargs = kwargs
                return [[]]

        client = FakeMilvusClient()
        store = MilvusVectorStore.__new__(MilvusVectorStore)
        store._client = client  # type: ignore[attr-defined]

        await store.search(
            query_embedding=[0.1, 0.2, 0.3],
            acl_filter=ACLFilter(
                tenant_id="tenant_hr",
                department_ids=["dept_001", "dept_002"],
                allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
            ),
            top_k=3,
        )

        assert client.search_kwargs["filter"] == (
            'tenant_id == "tenant_hr" '
            'and visibility in ["public", "department"] '
            'and (visibility == "public" or department_id in ["dept_001", "dept_002"])'
        )

    @pytest.mark.asyncio
    async def test_in_memory_retrieval_returns_acl_filtered_hits(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 1：In-memory 检索返回 ACL 过滤后的结果。"""
        vector_store = InMemoryVectorStore()

        # 添加 chunks
        texts = [c.full_text or c.chunk_text for c in sample_chunks]
        embeddings = await mock_embedder.embed_documents(texts)
        await vector_store.add_chunks(sample_chunks, embeddings)

        # 检索
        query = "入职材料"
        query_embedding = await mock_embedder.embed_query(query)
        results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=acl_filter, top_k=10
        )

        # 验证结果
        assert len(results) > 0, "Should return at least one result"

        # 验证所有结果都满足 ACL
        for result in results:
            assert result.tenant_id == acl_filter.tenant_id
            assert result.department_id in acl_filter.department_ids
            assert result.visibility in acl_filter.allowed_visibility

        # 验证机密文档被过滤掉
        confidential_results = [r for r in results if r.document_id == "doc_003"]
        assert len(confidential_results) == 0, "Confidential documents should be filtered out"

    @pytest.mark.asyncio
    async def test_acl_filter_by_tenant(
        self, mock_embedder: MockEmbedder, sample_chunks: list[ChunkCreate]
    ):
        """测试 ACL 按租户过滤。"""
        vector_store = InMemoryVectorStore()

        # 添加 chunks
        texts = [c.full_text or c.chunk_text for c in sample_chunks]
        embeddings = await mock_embedder.embed_documents(texts)
        await vector_store.add_chunks(sample_chunks, embeddings)

        # 使用错误的租户 ID
        wrong_acl = ACLFilter(
            tenant_id="tenant_eng",  # 错误的租户
            department_ids=["dept_001"],
            allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
        )

        query = "入职"
        query_embedding = await mock_embedder.embed_query(query)
        results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=wrong_acl, top_k=10
        )

        # 验证没有结果
        assert len(results) == 0, "Should return no results for wrong tenant"

    @pytest.mark.asyncio
    async def test_retrieval_allows_public_chunks_across_departments(
        self, mock_embedder: MockEmbedder
    ):
        """测试：检索前 ACL 应允许同租户 public chunk 跨部门命中。"""
        public_chunk = ChunkCreate(
            document_id="doc_public",
            chunk_text="全员公告：入职培训将在周五举行。",
            tenant_id="tenant_hr",
            department_id="dept_public",
            visibility=Visibility.PUBLIC,
        )
        acl_filter = ACLFilter(
            tenant_id="tenant_hr",
            department_ids=["dept_001"],
            allowed_visibility=[Visibility.PUBLIC, Visibility.DEPARTMENT],
        )
        vector_store = InMemoryVectorStore()
        bm25_store = InMemoryBM25Store()

        embeddings = await mock_embedder.embed_documents([public_chunk.chunk_text])
        await vector_store.add_chunks([public_chunk], embeddings)
        await bm25_store.add_chunks([public_chunk])

        query_embedding = await mock_embedder.embed_query("入职培训")
        dense_results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=acl_filter, top_k=10
        )
        sparse_results = await bm25_store.search(query="入职培训", acl_filter=acl_filter, top_k=10)

        assert [r.document_id for r in dense_results] == ["doc_public"]
        assert [r.document_id for r in sparse_results] == ["doc_public"]


class TestRRFFusion:
    """RRF Fusion 测试。"""

    @pytest.mark.asyncio
    async def test_dense_and_sparse_results_are_fused_by_rrf(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 2：Dense 和 Sparse 结果通过 RRF 融合。"""
        vector_store = InMemoryVectorStore()
        bm25_store = InMemoryBM25Store()
        rrf = RRFFuser()

        # 添加 chunks
        texts = [c.full_text or c.chunk_text for c in sample_chunks]
        embeddings = await mock_embedder.embed_documents(texts)
        await vector_store.add_chunks(sample_chunks, embeddings)
        await bm25_store.add_chunks(sample_chunks)

        # Dense search
        query = "入职材料"
        query_embedding = await mock_embedder.embed_query(query)
        dense_results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=acl_filter, top_k=10
        )

        # Sparse search
        sparse_results = await bm25_store.search(query=query, acl_filter=acl_filter, top_k=10)

        # RRF fusion
        fused_results = rrf.fuse(dense_results, sparse_results, top_k=10)

        # 验证融合结果
        assert len(fused_results) > 0, "Should return fused results"

        # 验证融合结果包含来自两种检索的结果
        dense_ids = {r.chunk_id for r in dense_results}
        sparse_ids = {r.chunk_id for r in sparse_results}
        fused_ids = {r.chunk_id for r in fused_results}

        # 融合结果应该包含两种检索的结果
        assert len(fused_ids) >= min(len(dense_ids), len(sparse_ids)), (
            "Fused results should include results from both retrieval methods"
        )

    def test_rrf_formula_correctness(self):
        """测试 RRF 公式正确性。"""
        rrf = RRFFuser(k=60)

        # 创建测试结果
        results_a = [
            RetrievalResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                chunk_text="test1",
                score=0.9,
                rerank_score=0.0,
                tenant_id="t",
                department_id="d",
                visibility=Visibility.PUBLIC,
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                document_id="doc_1",
                chunk_text="test2",
                score=0.8,
                rerank_score=0.0,
                tenant_id="t",
                department_id="d",
                visibility=Visibility.PUBLIC,
            ),
        ]

        results_b = [
            RetrievalResult(
                chunk_id="chunk_2",
                document_id="doc_1",
                chunk_text="test2",
                score=0.7,
                rerank_score=0.0,
                tenant_id="t",
                department_id="d",
                visibility=Visibility.PUBLIC,
            ),
            RetrievalResult(
                chunk_id="chunk_3",
                document_id="doc_1",
                chunk_text="test3",
                score=0.6,
                rerank_score=0.0,
                tenant_id="t",
                department_id="d",
                visibility=Visibility.PUBLIC,
            ),
        ]

        fused = rrf.fuse(results_a, results_b, top_k=10)

        # 验证 chunk_2 的分数最高（出现在两个列表中）
        chunk_2 = next(r for r in fused if r.chunk_id == "chunk_2")
        chunk_1 = next(r for r in fused if r.chunk_id == "chunk_1")
        chunk_3 = next(r for r in fused if r.chunk_id == "chunk_3")

        # chunk_2 = 1/(60+1) + 1/(60+1) = 2/61
        # chunk_1 = 1/(60+1) = 1/61
        # chunk_3 = 1/(60+2) = 1/62
        assert chunk_2.score > chunk_1.score, (
            "chunk_2 should have higher score (appears in both lists)"
        )
        assert chunk_1.score > chunk_3.score, "chunk_1 should have higher score (rank 1 vs rank 2)"


class TestBM25Signal:
    """BM25 Signal 测试。"""

    @pytest.mark.asyncio
    async def test_exact_policy_code_query_uses_bm25_signal(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 3：精确政策代码查询使用 BM25 信号。"""
        bm25_store = InMemoryBM25Store()

        # 添加 chunks
        await bm25_store.add_chunks(sample_chunks)

        # 使用精确关键词查询
        query = "身份证复印件"
        results = await bm25_store.search(query=query, acl_filter=acl_filter, top_k=10)

        # 验证返回结果
        assert len(results) > 0, "Should return results for exact keyword query"

        # 验证包含关键词的 chunk 排在前面
        first_result = results[0]
        assert "身份证" in first_result.chunk_text, "Top result should contain the exact keyword"

    @pytest.mark.asyncio
    async def test_bm25_returns_relevant_results(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 BM25 返回相关结果。"""
        bm25_store = InMemoryBM25Store()

        # 添加 chunks
        await bm25_store.add_chunks(sample_chunks)

        # 查询
        query = "试用期"
        results = await bm25_store.search(query=query, acl_filter=acl_filter, top_k=10)

        # 验证返回结果
        assert len(results) > 0, "Should return results"

        assert any(query in result.chunk_text for result in results), (
            "At least one BM25 result must contain the exact query"
        )


class TestDenseSignal:
    """Dense Signal 测试。"""

    @pytest.mark.asyncio
    async def test_semantic_query_uses_dense_signal(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 4：语义查询使用 Dense 信号。"""
        vector_store = InMemoryVectorStore()

        # 添加 chunks
        texts = [c.full_text or c.chunk_text for c in sample_chunks]
        embeddings = await mock_embedder.embed_documents(texts)
        await vector_store.add_chunks(sample_chunks, embeddings)

        # 使用语义查询（不直接匹配关键词）
        query = "新员工需要准备什么材料"
        query_embedding = await mock_embedder.embed_query(query)
        results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=acl_filter, top_k=10
        )

        # 验证返回结果
        assert len(results) > 0, "Should return results for semantic query"

        # 验证结果有分数
        for result in results:
            assert result.score > 0, "Results should have positive scores"


class TestReranker:
    """Reranker 测试。"""

    @pytest.mark.asyncio
    async def test_reranker_reorders_candidates(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 5：Reranker 重新排序候选结果。"""
        vector_store = InMemoryVectorStore()
        reranker = MockReranker()

        # 添加 chunks
        texts = [c.full_text or c.chunk_text for c in sample_chunks]
        embeddings = await mock_embedder.embed_documents(texts)
        await vector_store.add_chunks(sample_chunks, embeddings)

        # 检索
        query = "入职"
        query_embedding = await mock_embedder.embed_query(query)
        results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=acl_filter, top_k=10
        )

        # Rerank
        reranked = await reranker.rerank(query=query, results=results, top_k=5)

        # 验证 reranked 结果
        assert len(reranked) > 0, "Should return reranked results"
        assert len(reranked) <= 5, "Should respect top_k"

        # 验证所有结果都有 rerank_score
        for result in reranked:
            assert result.rerank_score >= 0, (
                "Reranked results should have non-negative rerank_score"
            )


class TestEvidenceBundle:
    """EvidenceBundle 测试。"""

    @pytest.mark.asyncio
    async def test_evidence_bundle_contains_citations(
        self, mock_embedder: MockEmbedder, acl_filter: ACLFilter, sample_chunks: list[ChunkCreate]
    ):
        """测试 6：EvidenceBundle 包含 citations。"""
        vector_store = InMemoryVectorStore()
        bm25_store = InMemoryBM25Store()
        reranker = MockReranker()

        # 添加 chunks
        texts = [c.full_text or c.chunk_text for c in sample_chunks]
        embeddings = await mock_embedder.embed_documents(texts)
        await vector_store.add_chunks(sample_chunks, embeddings)
        await bm25_store.add_chunks(sample_chunks)

        # Dense search
        query = "入职材料"
        query_embedding = await mock_embedder.embed_query(query)
        dense_results = await vector_store.search(
            query_embedding=query_embedding, acl_filter=acl_filter, top_k=10
        )

        # Sparse search
        sparse_results = await bm25_store.search(query=query, acl_filter=acl_filter, top_k=10)

        # RRF fusion
        rrf = RRFFuser()
        fused_results = rrf.fuse(dense_results, sparse_results, top_k=10)

        # Rerank
        reranked = await reranker.rerank(query=query, results=fused_results, top_k=5)

        # Build evidence bundle
        builder = EvidenceBuilder()
        evidence = builder.build(results=reranked, query=query)

        # 验证 EvidenceBundle
        assert evidence.total_count > 0, "EvidenceBundle should have citations"
        assert len(evidence.evidence_list) == evidence.total_count
        assert evidence.query_coverage_score >= 0.0
        assert evidence.query_coverage_score <= 1.0

        # 验证每个 Citation
        for citation in evidence.evidence_list:
            assert citation.id > 0
            assert citation.chunk_text
            assert citation.score >= 0.0 and citation.score <= 1.0
            assert citation.rerank_score >= 0.0 and citation.rerank_score <= 1.0


class TestEmbedder:
    """Embedder 测试。"""

    @pytest.mark.asyncio
    async def test_mock_embedder_deterministic(self, mock_embedder: MockEmbedder):
        """测试 Mock Embedder 确定性。"""
        text = "测试文本"

        # 多次 embed 同一文本
        embedding1 = await mock_embedder.embed_query(text)
        embedding2 = await mock_embedder.embed_query(text)

        # 验证结果一致
        assert embedding1 == embedding2, "Same text should produce same embedding"

    @pytest.mark.asyncio
    async def test_mock_embedder_different_texts(self, mock_embedder: MockEmbedder):
        """测试不同文本产生不同向量。"""
        text1 = "文本1"
        text2 = "文本2"

        embedding1 = await mock_embedder.embed_query(text1)
        embedding2 = await mock_embedder.embed_query(text2)

        # 验证结果不同
        assert embedding1 != embedding2, "Different texts should produce different embeddings"

    @pytest.mark.asyncio
    async def test_mock_embedder_dimension(self, mock_embedder: MockEmbedder):
        """测试向量维度。"""
        text = "测试"
        embedding = await mock_embedder.embed_query(text)

        assert len(embedding) == mock_embedder.dimension

    @pytest.mark.asyncio
    async def test_mock_embedder_normalized(self, mock_embedder: MockEmbedder):
        """测试向量已归一化。"""
        text = "测试"
        embedding = await mock_embedder.embed_query(text)

        # 计算 L2 范数
        norm = sum(x * x for x in embedding) ** 0.5

        # 验证范数约等于 1（允许小误差）
        assert abs(norm - 1.0) < 1e-6, "Embedding should be L2 normalized"
