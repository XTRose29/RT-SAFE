const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs');const path=require('path');const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');const data=JSON.parse(fs.readFileSync(path.join(root,'app/data.json')));
const matched=JSON.parse(fs.readFileSync(path.join(root,'app/matched-default-results.json')));
const subset=JSON.parse(fs.readFileSync(path.join(root,'app/subset-results.json')));
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}); const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 fs.mkdirSync(path.join(root,'nyc/previews/site-final'),{recursive:true});
 await page.goto(process.env.SITE_URL||'http://127.0.0.1:4173/',{waitUntil:'networkidle'});
 assert.equal(await page.locator('#demo').count(),1);
 assert(await page.evaluate(()=>document.querySelector('#demo').getBoundingClientRect().top<document.querySelector('h1').getBoundingClientRect().top));
 assert.equal(await page.locator('video').first().getAttribute('aria-label'),'RT-SAFE project film');
 await page.screenshot({path:path.join(root,'nyc/previews/site-final/demo-first.png')});
 const table=page.locator('.results-explorer');const effort=page.getByLabel('Leaderboard reasoning effort');
 const dataset=page.getByLabel('Leaderboard evaluation set');
 assert.equal(await dataset.inputValue(),'matched');
 assert.equal(await effort.count(),0);
 assert.equal(await table.locator('tbody tr').count(),9);
 assert.match(await table.locator('.results-context').textContent(),/4 maps · 20 tasks per model/);
 for(const score of matched.models){
  const row=table.locator('tbody tr').filter({has:page.getByRole('button',{name:score.fullName+' '+score.provider+' '+score.effort+' · Configured default',exact:true})});
  assert.deepEqual(await row.locator('td').allTextContents(),[`${score.success.toFixed(1)}%`,`${score.safeSuccess.toFixed(1)}%`,score.collisions.toFixed(2),score.spl.toFixed(3),`${score.latency.toFixed(1)} s`,score.decisions.toFixed(1)]);
 }
 const matchedCSV=decodeURIComponent((await table.locator('.csv-link').getAttribute('href')).split(',').slice(1).join(',')).split('\n');
 assert.equal(matchedCSV.length,10);
 assert(matchedCSV.slice(1).every(row=>row.includes('"hard-realtime-matched20-default-v1","20","4","0"')));
 await table.getByLabel('Filter models').fill('Gemini 4');
 assert.equal(await table.locator('tbody tr').count(),1);
 await table.locator('tbody button').click();
 assert.equal(await table.locator('.model-score strong').textContent(),matched.models.find(m=>m.name==='Gemini 4').collisions.toFixed(2));
 await table.getByLabel('Filter models').fill('');
 await table.screenshot({path:path.join(root,'nyc/previews/site-final/matched-twenty-leaderboard.png')});
 await page.locator('.matched-leaderboard-figure').screenshot({path:path.join(root,'nyc/previews/site-final/matched-twenty-figure.png')});
 await dataset.selectOption('subset');
 assert.equal(await dataset.inputValue(),'subset');
 assert.match(await table.locator('.results-context').textContent(),/2 maps · 10 tasks per configuration/);
 assert.equal(await table.getByRole('button',{name:'SPL',exact:false}).isEnabled(),true);
 for(const model of subset.models){
  const identity=data.models.find(m=>m.name===model.name);
  for(const [index,score] of model.effort.entries()){
   const label=score.label+(index===(model.name==='Sol'?0:1)?' · Provider default':'');
   const row=table.locator('tbody tr').filter({has:page.getByRole('button',{name:identity.fullName+' '+identity.provider+' '+label,exact:true})});
   assert.deepEqual(await row.locator('td').allTextContents(),[`${score.success.toFixed(1)}%`,`${score.safeSuccess.toFixed(1)}%`,score.collisions.toFixed(2),score.spl.toFixed(3),`${score.latency.toFixed(1)} s`,score.decisions.toFixed(1)]);
  }
 }
 const subtitleSizes=await table.locator('tbody tr').first().locator('th small').evaluateAll(nodes=>nodes.map(node=>parseFloat(getComputedStyle(node).fontSize)));
 assert(subtitleSizes.every(size=>size>=13));
 assert.equal(await table.locator('.model-compare').count(),0);
 const subsetCSV=decodeURIComponent((await table.locator('.csv-link').getAttribute('href')).split(',').slice(1).join(',')).split('\n');
 assert.equal(subsetCSV.length,25);
 assert(subsetCSV.slice(1).every(row=>row.includes('"hard-realtime-balanced-v1","10","2","0"')));
 await table.screenshot({path:path.join(root,'nyc/previews/site-final/balanced-leaderboard.png')});
 await table.getByRole('button',{name:'SPL',exact:false}).click();
 assert.equal(await table.locator('thead th').nth(4).getAttribute('aria-sort'),'descending');
 await dataset.selectOption('manuscript');
 assert.equal(await table.getByRole('button',{name:'SPL',exact:false}).isEnabled(),false);
 assert.equal(await table.locator('thead th').nth(3).getAttribute('aria-sort'),'ascending');
 assert.match(await table.locator('.results-context').textContent(),/36 episodes per configuration/);
 assert.equal(await effort.inputValue(),'all');
 assert.equal(await table.locator('tbody tr').count(),24);
 const initialCollisions=await table.locator('tbody tr td:nth-child(4)').allTextContents();
 assert.deepEqual(initialCollisions.map(Number),initialCollisions.map(Number).sort((a,b)=>a-b));
 const csvRows=decodeURIComponent((await table.locator('.csv-link').getAttribute('href')).split(',').slice(1).join(',')).split('\n');
 assert.equal(csvRows.length,25);
 assert.equal(new Set(csvRows.slice(1)).size,24);
 await table.getByLabel('Filter models').fill('Sol');
 assert.equal(await table.locator('tbody tr').count(),3);
 const solRows=table.locator('tbody tr');
 for(let i=0;i<3;i++){
  await solRows.nth(i).getByRole('button').click();
  assert.equal(await table.locator('.model-score strong').textContent(),await solRows.nth(i).locator('td').nth(2).textContent());
  assert.equal(await table.locator('tr.selected').count(),1);
 }
 await table.getByLabel('Filter models').fill('no-matching-model');
 assert.equal(await table.locator('.model-score').count(),0);
 await table.getByRole('button',{name:'Clear search',exact:true}).click();
 assert.equal(await table.locator('tbody tr').count(),24);
 await effort.selectOption('default');
 assert.equal(await table.locator('tbody tr').count(),8);
 for(const m of data.models){
  const row=table.locator('tbody tr').filter({has:page.getByRole('button',{name:m.fullName+' '+m.provider+' '+m.effort[m.name==='Sol'?0:1].label+' · Provider default',exact:true})});
  assert.equal(await row.count(),1);
 }
 const timing=page.locator('.nyc-timing-story');
 const timingBox=await timing.boundingBox();
 assert.equal(timingBox.width,960);
 assert(Math.abs(timingBox.x-(1440-timingBox.width)/2)<2);
 await timing.screenshot({path:path.join(root,'nyc/previews/site-final/smaller-timing-video.png')});
 for(let i=0;i<3;i++){
  await effort.selectOption(String(i));
  assert.equal(await page.getByRole('button',{name:'Hard · Real-time',exact:true}).getAttribute('aria-pressed'),'true');
  for(const m of data.models){
   const row=table.locator('tbody tr').filter({has:page.getByRole('button',{name:m.fullName+' '+m.provider+' '+m.effort[i].label+(i===(m.name==='Sol'?0:1)?' · Provider default':''),exact:true})});
   const cells=await row.locator('td').allTextContents();const e=m.effort[i];
   assert.deepEqual(cells,[`${e.success.toFixed(1)}%`,`${e.safeSuccess.toFixed(1)}%`,e.collisions.toFixed(2),'—',`${e.latency.toFixed(1)} s`,e.decisions.toFixed(1)],m.name+' effort '+i);
  }
  const csv=await table.locator('.csv-link').getAttribute('href');
  const rows=decodeURIComponent(csv.split(',').slice(1).join(',')).split('\n');assert.equal(rows.length,9);
  assert(rows.slice(1).every(x=>x.includes('"realtime"')));
 }
 await table.getByRole('button',{name:'Success',exact:false}).first().click();
 assert.equal(await table.locator('thead th').nth(1).getAttribute('aria-sort'),'descending');
 await table.getByLabel('Filter models').fill('Sol');assert.equal(await table.locator('tbody tr').count(),1);
 await table.locator('tbody button').click();assert.equal(await table.locator('.model-score strong').textContent(),'31.50');
 await table.getByLabel('Filter models').fill('');
 await table.screenshot({path:path.join(root,'nyc/previews/site-final/effort-leaderboard.png')});
 await page.getByRole('button',{name:'All difficulties',exact:true}).click();assert.equal(await effort.count(),0);
 assert.equal(await table.locator('tbody tr').count(),8);
 for(const name of ['Astra','Sol','Sonnet','Fable','Gemini','DeepSeek','Inkling','Grok']){
  await page.getByRole('group',{name:'Choose a model example'}).getByRole('button',{name,exact:true}).click();
  const card=page.locator('.behavior-card');
  await card.getByRole('button',{name:/More turning:/}).locator('text').click();
  assert.equal(await card.locator('.radar-readout>span').textContent(),'Turn actions');
  await card.locator('summary').click();assert.equal(await card.locator('.profile-details dl>div').count(),10);
  await card.locator('summary').click();
 }
 await page.getByRole('group',{name:'Choose a model example'}).getByRole('button',{name:'Astra',exact:true}).click();
 const video=page.locator('.model-overview video');await video.evaluate(v=>v.play());await page.waitForTimeout(500);
 assert(await video.evaluate(v=>v.currentTime>0 && v.muted));await video.evaluate(v=>v.pause());
 await page.locator('.model-overview').screenshot({path:path.join(root,'nyc/previews/site-final/interactive-astra.png')});
 assert.equal(await page.locator('video track').count(),0);assert(await page.locator('video').evaluateAll(vs=>vs.every(v=>v.muted)));
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.setViewportSize({width:390,height:844});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.getByRole('button',{name:'Hard · Real-time',exact:true}).click();assert.equal(await effort.inputValue(),'all');assert.equal(await table.locator('tbody tr').count(),24);await dataset.selectOption('subset');
 const mobileSizes=await table.locator('tbody tr').first().locator('th small').evaluateAll(nodes=>nodes.map(node=>parseFloat(getComputedStyle(node).fontSize)));
 assert(mobileSizes.every(size=>size>=12));
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await table.screenshot({path:path.join(root,'nyc/previews/site-final/mobile-effort.png')});
 await dataset.selectOption('matched');
 assert.equal(await table.locator('tbody tr').count(),9);
 assert.equal(await effort.count(),0);
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await table.screenshot({path:path.join(root,'nyc/previews/site-final/mobile-matched-twenty.png')});
 await page.locator('.behavior-card').first().screenshot({path:path.join(root,'nyc/previews/site-final/mobile-interactive-radar.png')});
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:path.join(root,'nyc/previews/site-final/mobile-demo-first.png')});
 assert.deepEqual(errors,[]);
 const result={matched_default_rows_verified:9,matched_episodes:180,matched_csv:true,effort_rows_verified:48,subset_episodes:240,larger_provider_and_effort_labels:true,all_efforts_default:true,unique_configuration_selection:true,smaller_timing_video:true,scoped_csv:true,sort_search_selection:true,interactive_profiles:8,astra_replay:true,demo_first:true,mobile_overflow:false,all_muted:true,browser_errors:errors};
 fs.writeFileSync(path.join(root,'nyc/previews/site-final/benchmark-validation.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result,null,2));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
