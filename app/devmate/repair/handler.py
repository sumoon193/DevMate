"""修复计划 typed 入口。

合同：``RuntimeEvent.execute(input: DM08Input) -> DM08Result``。
RepairPlan 只生成不可变 patch artifact。
"""

from __future__ import annotations

from app.devmate.repair.plan import RepairPlan
from app.devmate.repair.types import DM08Input, DM08Result


class RuntimeEvent:
    def __init__(self, planner: RepairPlan | None = None) -> None:
        self.planner = planner or RepairPlan()

    def execute(self, input_: DM08Input) -> DM08Result:
        return self.planner.create(input_)
