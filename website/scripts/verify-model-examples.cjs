/* Check all eight recorded players, seek controls, profile links, and mobile layout. */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs'),path=require('path');
const data=require('../app/model-examples.json');
const out=process.env.CHECK_OUTPUT||'/tmp/rt-safe-model-examples';fs.mkdirSync(out,{recursive:true});
const assert=(ok,msg)=>{if(!ok)throw Error(msg)};
(async()=>{
 const b=await chromium.launch({headless:true,args:['--no-sandbox']});
 const p=await b.newPage({viewport:{width:1440,height:1100}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
 await p.goto(process.env.SITE_URL||'http://127.0.0.1:4187/',{waitUntil:'networkidle'});
 const root=p.locator('.model-examples'),selector=root.getByRole('group',{name:'Choose a model example'});
 assert(await selector.locator('button').count()===8,'Missing model selector');
 for(const model of data.models){
  await selector.getByRole('button',{name:model.name,exact:true}).click();
  const v=root.locator('video');await v.evaluate(v=>v.play());
  await p.waitForFunction(()=>document.querySelector('.model-examples video')?.currentTime>.1);
  const info=await v.evaluate(v=>({src:v.currentSrc,duration:v.duration,muted:v.muted,tracks:v.textTracks.length,w:v.videoWidth,h:v.videoHeight}));
  assert(info.src.endsWith(model.video),`${model.name}: wrong footage`);
  assert(Math.abs(info.duration-model.videoDuration)<.05,`${model.name}: wrong duration`);
  assert(info.muted&&info.tracks===0&&info.w===960&&info.h===1000,`${model.name}: stream properties`);
  await root.locator('.example-chapters button').nth(1).click();
  await p.waitForTimeout(180);
  assert(Math.abs((await v.evaluate(v=>v.currentTime))-model.steps[1].start/model.speed)<1,`${model.name}: chapter seek failed`);
  if(model.events.length){await root.getByRole('button',{name:'Jump to first collision report'}).click();await p.waitForTimeout(180);assert(Math.abs((await v.evaluate(v=>v.currentTime))-Math.max(0,model.events[0].at-.3))<1,'Contact seek failed');}
  await v.evaluate(v=>v.pause());
 }
 await selector.getByRole('button',{name:'Astra',exact:true}).click();
 await root.screenshot({path:path.join(out,'desktop.png')});
 await root.getByRole('button',{name:/Compare Astra & Sol/}).click();assert(await root.locator('.nyc-case video').count()===2,'Original comparison missing');
 await selector.getByRole('button',{name:'Fable',exact:true}).click();assert(await root.locator('video').count()===1,'Comparison players not removed');
 assert((await root.locator('.example-other-events').innerText()).includes('1 hazard interaction'),'Fable hazard missing');
 for(const model of data.models){
  const card=p.locator('.behavior-card').filter({has:p.locator('.behavior-model',{hasText:new RegExp('^'+model.name+'$')})});
  await card.getByRole('button',{name:`▶ Watch ${model.name} example`,exact:true}).click();
  assert(await card.locator('video source').getAttribute('src')===model.video,`${model.name}: wrong profile replay`);
  await card.getByRole('button',{name:'Close recorded example',exact:true}).click();assert(await card.locator('video').count()===0,'Profile video not removed');
 }
 await p.setViewportSize({width:390,height:844});await selector.getByRole('button',{name:'Fable',exact:true}).focus();await root.scrollIntoViewIfNeeded();
 assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile overflow');
 await root.screenshot({path:path.join(out,'mobile.png')});
 assert(errors.length===0,JSON.stringify(errors));
 console.log('All eight players, decision/contact seeks, profile examples, comparison and mobile checks passed.');await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
