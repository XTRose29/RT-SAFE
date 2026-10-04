"""Export the manuscript's six-axis behavior profiles as standalone plots."""
from pathlib import Path
import json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'public/media/behavior';OUT.mkdir(parents=True,exist_ok=True)
fontManager.addfont(ROOT/'public/fonts/space-grotesk-500.ttf');plt.rcParams['font.family']='Space Grotesk'
DATA=json.loads((ROOT/'app/behavior.json').read_text())
COLORS={'Inkling':'#518b37','Grok':'#c23b52','Astra':'#16878a','Fable':'#9557b4','Sol':'#2582b4','Sonnet':'#a64f81','Gemini':'#b78917','DeepSeek':'#c2672b'}
labels=['Fewer\ncollisions','Quicker\ndecisions','Fewer\ndecisions','Longer commanded\nmoves','More\nwaiting','More\nturning']
angles=np.linspace(0,2*np.pi,6,endpoint=False);closed=np.r_[angles,angles[0]]
for profile in DATA['profiles']:
 fig=plt.figure(figsize=(5,5),facecolor='none');ax=fig.add_axes([.20,.19,.60,.60],polar=True)
 ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1);ax.set_ylim(0,1);ax.set_yticks([.25,.5,.75,1]);ax.set_yticklabels([])
 ax.set_xticks(angles);ax.set_xticklabels(labels,fontsize=15,color='#34495e');ax.tick_params(axis='x',pad=15)
 ax.grid(color='#dce4ec',linewidth=1);ax.spines['polar'].set_visible(False);ax.set_facecolor('none')
 values=np.r_[profile['values'],profile['values'][0]];col=COLORS[profile['name']]
 ax.plot(closed,values,color=col,linewidth=2.8);ax.fill(closed,values,color=col,alpha=.20)
 for suffix in ['png','svg']:fig.savefig(OUT/(profile['name'].lower()+'.'+suffix),dpi=180,transparent=True)
 svg=OUT/(profile['name'].lower()+'.svg')
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)
print('Exported 8 radar profiles as PNG and SVG')
