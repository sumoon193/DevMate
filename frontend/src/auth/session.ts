import Keycloak from 'keycloak-js'

const url=import.meta.env.VITE_KEYCLOAK_URL as string|undefined
const realm=import.meta.env.VITE_KEYCLOAK_REALM as string|undefined
const clientId=import.meta.env.VITE_KEYCLOAK_CLIENT_ID as string|undefined
const client=url&&realm&&clientId?new Keycloak({url,realm,clientId}):null
let token:string|undefined
export const accessToken=()=>token
export const oidcConfigured=()=>Boolean(client)
export async function initializeAuth(){if(!client)return;await client.init({onLoad:'login-required',pkceMethod:'S256',checkLoginIframe:false});token=client.token;window.setInterval(async()=>{if(await client.updateToken(45))token=client.token},30000)}
