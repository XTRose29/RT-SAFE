/* Browser checks for the native panorama explorer and silent video UI. */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path');
const out=path.resolve(__dirname,'../nyc/previews/explorer-check');fs.mkdirSync(out,{recursive:true});
const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
(async()=>{
 const b=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
 const p=await b.newPage({viewport:{width:1440,height:1100}}),errors=[];
 p.on('pageerror',e=>errors.push(e.message));p.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
 const url=process.env.SITE_URL||'http://127.0.0.1:4173/';await p.goto(url,{waitUntil:'networkidle'});
 assert(await p.locator('video track').count()===0,'Video subtitle tracks remain');
 assert((await p.locator('a[href*="captioned"],a[href$=".vtt"]').count())===0,'Captioned video links remain');
 assert(await p.locator('video').evaluateAll(vs=>vs.every(v=>v.muted)),'Some player is not muted');
 const root=p.locator('.environment-explorer'),c=root.locator('canvas');await root.scrollIntoViewIfNeeded();
 await p.waitForFunction(()=>document.querySelectorAll('.explorer-destinations button').length===4);
 await p.waitForTimeout(80);await root.locator('.explorer-status').waitFor({state:'hidden',timeout:30000});
 const capture=()=>c.screenshot();const first=await capture();
 const box=await c.boundingBox();await p.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await p.mouse.down();await p.mouse.move(box.x+box.width*.75,box.y+box.height*.55,{steps:12});await p.mouse.up();
 assert(!first.equals(await capture()),'Drag did not change camera');
 await root.getByRole('button',{name:'Reset view',exact:true}).click();await p.waitForTimeout(100);
 assert(first.equals(await capture()),'Reset did not restore camera');
 await c.focus();await c.press('ArrowRight');assert(!first.equals(await capture()),'Keyboard look failed');await root.getByRole('button',{name:'Reset view',exact:true}).click();
 await root.getByRole('button',{name:'Zoom in',exact:true}).click();assert(!first.equals(await capture()),'Zoom failed');await root.getByRole('button',{name:'Reset view',exact:true}).click();
 await root.getByRole('button',{name:'Auto-rotate'}).click();const start=await capture();await p.waitForTimeout(600);assert(!start.equals(await capture()),'Rotation did not advance');
 await root.getByRole('button',{name:'Pause rotation'}).click();const paused=await capture();await p.waitForTimeout(400);assert(paused.equals(await capture()),'Rotation did not pause');
 for(const label of ['Along the path','The crossing','In the park','The approach']) {
  await root.locator('.explorer-destinations').getByRole('button',{name:new RegExp(label)}).click();await p.waitForTimeout(80);await root.locator('.explorer-status').waitFor({state:'hidden',timeout:30000});
  assert((await root.locator('.explorer-location').innerText()).includes(label),'Wrong viewpoint');
  await root.locator('.explorer-viewport').screenshot({path:path.join(out,label.toLowerCase().replaceAll(' ','-')+'.png')});
 }
 for(const label of ['Collisions','Hazards','Traffic rules']) {
  await root.locator('.explorer-topics').getByRole('button',{name:new RegExp(label)}).click();await p.waitForTimeout(80);await root.locator('.explorer-status').waitFor({state:'hidden',timeout:30000});
  assert(await root.locator('.explorer-insight').isVisible(),'Safety detail missing');assert(await root.locator('.explorer-hotspot').isVisible(),'Safety anchor not visible');
  await root.locator('.explorer-viewport').screenshot({path:path.join(out,label.toLowerCase().replaceAll(' ','-')+'.png')});
 }
 // Rapid navigation must leave the last selected location visible.
 const buttons=root.locator('.explorer-destinations button');await buttons.nth(0).click();await buttons.nth(3).click();await buttons.nth(1).click();await p.waitForTimeout(80);await root.locator('.explorer-status').waitFor({state:'hidden',timeout:30000});assert((await root.locator('.explorer-location').innerText()).includes('Along the path'),'Rapid navigation race');
 await root.getByRole('button',{name:'Watch moving scene'}).click();const v=root.locator('video');await p.waitForFunction(()=>{const v=document.querySelector('.explorer-scene-video');return v.currentTime>.3&&!v.paused;});
 assert(Math.abs(await v.evaluate(v=>v.duration)-12)<.05,'Unexpected scene duration');await root.getByRole('button',{name:'Pause scene',exact:true}).click();const time=await v.evaluate(v=>v.currentTime);await p.waitForTimeout(350);assert(Math.abs(await v.evaluate(v=>v.currentTime)-time)<.05,'Scene did not pause');
 await root.getByRole('button',{name:'Return to 360°'}).click();assert(await v.evaluate(v=>v.paused),'Hidden video still playing');
 await root.getByRole('button',{name:'Toggle environment fullscreen'}).click();await p.waitForFunction(()=>!!document.fullscreenElement); await p.evaluate(()=>document.exitFullscreen());
 await root.screenshot({path:path.join(out,'desktop.png')});
 assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Desktop overflow');
 await p.setViewportSize({width:390,height:844});await root.scrollIntoViewIfNeeded();assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile overflow');
 await root.screenshot({path:path.join(out,'mobile.png')});
 // Browser-native touch events exercise the same drag controller on mobile.
 const bb=await c.boundingBox(),session=await p.context().newCDPSession(p);await session.send('Emulation.setTouchEmulationEnabled',{enabled:true});const beforeTouch=await capture();
 await session.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:bb.x+bb.width*.4,y:bb.y+bb.height*.3}]});
 await session.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:bb.x+bb.width*.7,y:bb.y+bb.height*.35}]});
 await session.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});assert(!beforeTouch.equals(await capture()),'Touch drag failed');
 assert(errors.length===0,JSON.stringify(errors));
 fs.writeFileSync(path.join(out,'validation.json'),JSON.stringify({url,drag:true,keyboard:true,zoom:true,rotation_pause:true,viewpoints:4,safety_details:3,rapid_navigation:true,video_play_pause:true,fullscreen:true,touch_drag:true,no_overflow:true,no_caption_tracks:true,all_players_muted:true,errors},null,2));
 console.log('Explorer checks passed.');await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
