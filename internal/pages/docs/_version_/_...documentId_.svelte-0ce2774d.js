import{S as L,i as P,s as B,k as y,l as E,a8 as N,d as f,m as x,g as _,o as h,p as O,q as v,r as R,aw as U,n as F,e as p,w as A,c as b,a as k,x as T,b as g,J as I,y as j,B as S,t as C,h as V,j as G,K as q}from"../../../chunks/vendor-f09d08fb.js";import{R as H,D as J,e as K,S as M}from"../../../chunks/config-a985ffc1.js";import{H as $,b as z,a as Q}from"../../../chunks/documentloader-a696cf7c.js";const{document:w}=U;function W(d){let a,n,e,r,o,l,s,u;return e=new z({props:{documentId:d[1],loader:d[2]}}),s=new Q({props:{document:d[3],views:M}}),{c(){a=p("div"),n=p("div"),A(e.$$.fragment),r=y(),o=p("div"),l=p("div"),A(s.$$.fragment),this.h()},l(c){a=b(c,"DIV",{class:!0});var t=k(a);n=b(t,"DIV",{class:!0});var i=k(n);T(e.$$.fragment,i),i.forEach(f),r=x(t),o=b(t,"DIV",{class:!0});var m=k(o);l=b(m,"DIV",{class:!0});var D=k(l);T(s.$$.fragment,D),D.forEach(f),m.forEach(f),t.forEach(f),this.h()},h(){g(n,"class","gutter"),g(l,"class","document "+d[3].tag),g(o,"class","content h-max p-4 sm:w-full md:max-w-2xl"),g(a,"class","flex lg:flex-row flex-col lg:sticky lg:h-full")},m(c,t){_(c,a,t),I(a,n),j(e,n,null),I(a,r),I(a,o),I(o,l),j(s,l,null),u=!0},p(c,t){const i={};t&2&&(i.documentId=c[1]),t&4&&(i.loader=c[2]),e.$set(i)},i(c){u||(v(e.$$.fragment,c),v(s.$$.fragment,c),u=!0)},o(c){h(e.$$.fragment,c),h(s.$$.fragment,c),u=!1},d(c){c&&f(a),S(e),S(s)}}}function X(d){let a,n;return{c(){a=C("An error occured :( : "),n=C(d[0])},l(e){a=V(e,"An error occured :( : "),n=V(e,d[0])},m(e,r){_(e,a,r),_(e,n,r)},p(e,r){r&1&&G(n,e[0])},i:q,o:q,d(e){e&&f(a),e&&f(n)}}}function Y(d){let a,n,e,r,o,l;w.title=a=H;const s=[X,W],u=[];function c(t,i){return t[0]?0:1}return e=c(d),r=u[e]=s[e](d),{c(){n=y(),r.c(),o=E()},l(t){N('[data-svelte="svelte-8a7s9"]',w.head).forEach(f),n=x(t),r.l(t),o=E()},m(t,i){_(t,n,i),u[e].m(t,i),_(t,o,i),l=!0},p(t,[i]){(!l||i&0)&&a!==(a=H)&&(w.title=a);let m=e;e=c(t),e===m?u[e].p(t,i):(F(),h(u[m],1,1,()=>{u[m]=null}),O(),r=u[e],r?r.p(t,i):(r=u[e]=s[e](t),r.c()),v(r,1),r.m(o.parentNode,o))},i(t){l||(v(r),l=!0)},o(t){h(r),l=!1},d(t){t&&f(n),u[e].d(t),t&&f(o)}}}const re=!0,oe=!0,ne=!1;async function se({params:d,fetch:a}){let{version:n,documentId:e}=d;e=e||J;const r=await a("/config").then(s=>s.json()),o=new $(r.basePath,n);o.fetch=a,o.attributes=await o.load("attributes"),o.load("linktree");let l={documentId:e,loader:o};return await o.load(e).then(s=>(l.error=!1,{props:l})).catch(s=>(l.error=s,{props:l}))}function Z(d,a,n){let{error:e}=a,{documentId:r}=a,{loader:o}=a;const l=o.cache[r];return R(K,o),d.$$set=s=>{"error"in s&&n(0,e=s.error),"documentId"in s&&n(1,r=s.documentId),"loader"in s&&n(2,o=s.loader)},[e,r,o,l]}class le extends L{constructor(a){super();P(this,a,Z,Y,B,{error:0,documentId:1,loader:2})}}export{le as default,oe as hydrate,se as load,re as prerender,ne as router};