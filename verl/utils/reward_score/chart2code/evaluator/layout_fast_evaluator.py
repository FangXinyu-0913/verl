# layout_fast.py
import ast, sys, io, textwrap, concurrent.futures as cf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PREFIX = textwrap.dedent("""
    import warnings, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
""")
def _extract_layout(fig):
    li=[]
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None: continue
        gs = spec.get_gridspec(); nrows,ncols = gs.get_geometry()
        li.append( dict(nrows=nrows,ncols=ncols,
                        row_start=spec.rowspan.start,
                        row_end=spec.rowspan.stop-1,
                        col_start=spec.colspan.start,
                        col_end=spec.colspan.stop-1) )
    return li

def _run(code:str)->list[dict]:
    glo={"plt":plt,"__name__":"__main__","_extract_layout":_extract_layout}
    exec(compile(_PREFIX+code,"<gen>","exec"), glo)
    fig=plt.gcf(); res=_extract_layout(fig); plt.close(fig); return res

class FastLayoutEvaluator:
    def __init__(self, pool:cf.ProcessPoolExecutor|None=None):
        self.metrics={"precision":0,"recall":0,"f1":0}
        self._pool = pool or cf.ProcessPoolExecutor(max_workers=4,
                                                    initializer=lambda: None)

    def __call__(self, generation_code_file, golden_code_file):
        with open(generation_code_file) as f: gen=f.read()
        with open(golden_code_file)   as f: gold=f.read()

        fut_g = self._pool.submit(_run, gen)
        fut_t = self._pool.submit(_run, gold)
        gen_lay, gold_lay = fut_g.result(), fut_t.result()
        self._calc(gen_lay, gold_lay)

    def _calc(self, g, d):
        if not g or not d: return
        gs, ds = set(map(tuple, g)), set(map(tuple, d))
        inter=len(gs & ds); p=inter/len(gs); r=inter/len(ds)
        self.metrics.update(precision=p,recall=r,
            f1=0 if p+r==0 else 2*p*r/(p+r))
