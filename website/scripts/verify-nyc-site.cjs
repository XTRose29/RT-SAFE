const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('fs');
const path = require('path');
const out=path.resolve(__dirname,'../nyc/previews/site-final');fs.mkdirSync(out,{recursive:true});
const assert=(ok,message)=>{if(!ok)throw new Error(message)};
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
 await page.goto(process.env.SITE_URL||'http://127.0.0.1:4173/',{waitUntil:'networkidle'});
 assert(await page.locator('.nyc-hero-video').count()===1,'NYC edition not loaded');
 await page.locator('.nyc-hero-video').evaluate(v=>new Promise((resolve,reject)=>{if(v.readyState>=2)resolve();else{v.addEventListener('loadeddata',resolve,{once:true});setTimeout(()=>reject(new Error('Hero timeout')),15000)}}));
 await page.screenshot({path:path.join(out,'desktop-top.png')});
 const results={};
 for(const [kind,duration] of [['timing',11],['case',44.1]]){
  const panel=page.locator(`.nyc-${kind}`);await panel.scrollIntoViewIfNeeded();
  await panel.locator('.nyc-play').click();await page.waitForTimeout(3500);
  let states=await panel.locator('video').evaluateAll(vs=>vs.map(v=>({t:v.currentTime,duration:v.duration,paused:v.paused,ready:v.readyState})));
  assert(states.every(v=>v.t>1&&!v.paused),`${kind} playback failed`);
  assert(Math.abs(states[0].t-states[1].t)<.2,`${kind} views drifted`);
  assert(states.every(v=>Math.abs(v.duration-duration)<.04),`${kind} duration wrong`);
  await panel.locator('.nyc-play').click();
  const slider=panel.locator('input[type=range]');await slider.focus();await slider.press('End');
  await page.waitForTimeout(400);
  states=await panel.locator('video').evaluateAll(vs=>vs.map(v=>({t:v.currentTime,paused:v.paused})));
  assert(states.every(v=>v.t>duration-.12&&v.paused),`${kind} end seek failed`);
  await panel.locator('.nyc-play').click();await page.waitForTimeout(800);
  states=await panel.locator('video').evaluateAll(vs=>vs.map(v=>({t:v.currentTime,paused:v.paused})));
  assert(states.every(v=>v.t<2&&!v.paused),`${kind} replay failed`);
  await panel.locator('.nyc-play').click();
  await slider.focus();for(let i=0;i<100;i++)await slider.press('ArrowRight');
  await panel.screenshot({path:path.join(out,`${kind}.png`)});results[kind]={synchronized_playback:true,seek:true,replay:true,duration};
 }
 await page.locator('.nyc-gallery summary').click();
 for(const button of await page.locator('.nyc-camera-tabs button').all()){
  await button.click();await page.locator('.nyc-gallery img').evaluate(im=>im.decode());
 }
 await page.locator('.nyc-gallery summary').click();
 const tour=page.locator('.nyc-environment video');await tour.evaluate(v=>{v.load()});
 await tour.evaluate(v=>new Promise((resolve,reject)=>{if(v.readyState>=1){resolve();return;}v.addEventListener('loadedmetadata',resolve,{once:true});setTimeout(()=>reject(new Error('Tour metadata timeout')),15000)}));
 assert(Math.abs(await tour.evaluate(v=>v.duration)-12)<.04,'Tour duration');
 await tour.evaluate(v=>{v.currentTime=6});await page.waitForTimeout(600);
 await page.locator('.nyc-environment').screenshot({path:path.join(out,'environment.png')});
 for(const [label,duration] of [['Astra vs. Sol · 0:44',44.1],['NYC project film · 1:30',90]]){
  await page.getByRole('button',{name:label,exact:true}).click();
  const film=page.locator('.film-shell video');await film.evaluate(v=>{v.load()});
  await film.evaluate(v=>new Promise((resolve,reject)=>{if(v.readyState>=1){resolve();return;}v.addEventListener('loadedmetadata',resolve,{once:true});setTimeout(()=>reject(new Error('Film metadata timeout')),15000)}));
  assert(Math.abs(await film.evaluate(v=>v.duration)-duration)<.04,'Film duration');
 }
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Desktop overflow');
 await page.setViewportSize({width:390,height:844});await page.goto(process.env.SITE_URL||'http://127.0.0.1:4173/',{waitUntil:'networkidle'});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile overflow');
 await page.screenshot({path:path.join(out,'mobile-top.png')});
 await page.locator('.nyc-case').screenshot({path:path.join(out,'mobile-comparison.png')});
 await page.emulateMedia({reducedMotion:'reduce'});
 await page.reload({waitUntil:'networkidle'});
 assert(await page.locator('.nyc-hero-video').evaluate(v=>v.paused),'Reduced-motion preference ignored');
 results.reduced_motion=true;
 assert(errors.length===0,errors.join('\n'));
 results.browser_errors=errors;results.desktop_overflow=false;results.mobile_overflow=false;results.gallery_cameras=4;
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify(results,null,2));console.log(JSON.stringify(results,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
