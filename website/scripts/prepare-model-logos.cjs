/* Rasterize the vendored SVG marks for the video renderer. */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('fs');const path=require('path');
(async()=>{const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const page=await browser.newPage({viewport:{width:256,height:256}});
for(const name of ['openai','claude','gemini','deepseek','grok']){const dir=path.resolve(__dirname,'../public/media/logos');let svg=fs.readFileSync(path.join(dir,name+'.svg'),'utf8');await page.setContent(`<style>html,body{margin:0;background:transparent}svg{width:256px;height:256px;color:#172b42}</style>${svg}`);await page.screenshot({path:path.join(dir,name+'.png'),omitBackground:true});}
await browser.close();})();
