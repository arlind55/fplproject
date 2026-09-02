const $=id=>document.getElementById(id);
const SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function currentTheme(){const t=document.documentElement.getAttribute("data-theme");if(t)return t;return matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";}
function applyTheme(t){if(t)document.documentElement.setAttribute("data-theme",t);else document.documentElement.removeAttribute("data-theme");$("themeBtn").innerHTML=currentTheme()==="dark"?SUN:MOON;}
try{applyTheme(localStorage.getItem("fpl-theme")||null);}catch(e){applyTheme(null);}
$("themeBtn").addEventListener("click",()=>{const next=currentTheme()==="dark"?"light":"dark";try{localStorage.setItem("fpl-theme",next);}catch(e){}applyTheme(next);});
