# text_fast.py -----------------------------------------------------------
import concurrent.futures as cf, textwrap, pathlib, gc, matplotlib
from collections import Counter
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import RendererPdf

_PREFIX = textwrap.dedent("""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
drawed_texts=[]
def _hook(func):
    def w(*a,**kw):
        obj,x,y,s=a[0],a[2],a[3],a[4]
        xr=x/obj.width/72*100; yr=y/obj.height/72*100
        drawed_texts.append((x,y,xr,yr,s)); return func(*a,**kw)
    return w

# Import and hook multiple renderers for different output formats
from matplotlib.backends.backend_pdf import RendererPdf
from matplotlib.backends.backend_agg import RendererAgg
try:
    from matplotlib.backends.backend_svg import RendererSVG
except ImportError:
    RendererSVG = None
try:
    from matplotlib.backends.backend_ps import RendererPS
except ImportError:
    RendererPS = None

RendererPdf.draw_text=_hook(RendererPdf.draw_text)  # PDF format
RendererAgg.draw_text=_hook(RendererAgg.draw_text)  # PNG, JPG, etc.
if RendererSVG is not None:
    RendererSVG.draw_text=_hook(RendererSVG.draw_text)  # SVG format
if RendererPS is not None:
    RendererPS.draw_text=_hook(RendererPS.draw_text)  # PS/EPS format
""")

_AXS_DEL = "for ax in plt.gcf().get_axes(): ax.set_xticks([]); ax.set_yticks([]);"

def _exec(code:str, rm_ticks:bool)->list[tuple]:
    if rm_ticks: code = code.replace("plt.savefig", _AXS_DEL+"plt.savefig")
    glob={"plt":plt,"__name__":"__main__","drawed_texts":[]}
    exec(compile(_PREFIX+code,"<gen>","exec"), glob)
    res=glob["drawed_texts"]; plt.close("all"); gc.collect(); return res

def _metric(gen, gold, use_pos):
    if not gen or not gold: return 0,0,0
    if not use_pos:
        g,d = Counter(t[-1] for t in gen), Counter(t[-1] for t in gold)
    else:
        q=lambda t:(round(t[2]/10),round(t[3]/10),t[-1])
        g,d = Counter(map(q,gen)),Counter(map(q,gold))
    inter=sum((g&d).values()); p=inter/len(gen); r=inter/len(gold)
    return p,r,0 if p+r==0 else 2*p*r/(p+r)

class FastTextEvaluator:
    def __init__(self,use_position=False,use_axs=True,pool=None):
        self.use_position=use_position; self.use_axs=use_axs
        self.pool=pool or cf.ProcessPoolExecutor(max_workers=4)
        self.metrics={"precision":0,"recall":0,"f1":0}

    def __call__(self,generation_code_file,golden_code_file):
        gcode=pathlib.Path(generation_code_file).read_text()
        dcode=pathlib.Path(golden_code_file).read_text()
        fut1=self.pool.submit(_exec,gcode,not self.use_axs)
        fut2=self.pool.submit(_exec,dcode,not self.use_axs)
        gen_txt,fut = fut1.result(), fut2.result()
        p,r,f1=_metric(gen_txt,fut,self.use_position)
        self.metrics.update(precision=p,recall=r,f1=f1)
