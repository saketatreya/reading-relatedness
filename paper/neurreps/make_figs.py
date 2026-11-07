"""Generate manuscript numbers, tables, and figures from the analysis result record."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / '../../results/analysis.json'
R = json.loads(SOURCE.read_text())
assert R['schema_version'] == 2 and R['provenance']['nperm'] == 2000
OUT, FIG = HERE / 'generated', HERE / 'figs'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
COLORS = {'euclid': '#3974a4', 'cosine': '#c06932'}
NAMES = {'text': 'Text', 'speech': 'Speech', 'protein': 'Protein', 'vision': 'Vision', 'singlecell': 'Single-cell'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'axes.labelsize': 10, 'legend.fontsize': 9, 'savefig.bbox': 'tight'})

def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out

def syn(cell, metric, *keys):
    vals = []
    for s in R['synthetic']['seeds']:
        x = s['cells'][cell]['metrics'][metric]
        for k in keys:
            x = x[k]
        vals.append(x)
    return np.array(vals, dtype=float)

def fmt(x, key=''):
    return str(x) if isinstance(x, int) else f'{x:.4f}' if key.endswith(('_p', 'p_taxon')) else f'{x:.3f}'

def table(name, columns, header, rows):
    lines = [r'\begin{tabular}{' + columns + '}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / f'{name}.tex').write_text('\n'.join(lines) + '\n')

PATHS = [('distance_r',), ('distance_quartet',), ('nj', 'split_recall'), ('fastme', 'split_recall'),
         ('nj', 'tree_quartet'), ('fastme', 'tree_quartet'), ('nj', 'deep_split_recall')]

def numbers():
    vals = flatten(R)
    q = [x['distance_quartet'] for d in R['domains'].values() for x in d['metrics'].values()]
    vals.update({'summary.quartet_min': min(q), 'summary.quartet_max': max(q)})
    for cell in R['synthetic']['seeds'][0]['cells']:
        for m in COLORS:
            for path in PATHS:
                a = syn(cell, m, *path)
                key = '.'.join(('synthetic_mean', cell, m) + path)
                vals[key], vals[key+'.sd'] = float(a.mean()), float(a.std(ddof=1))
    header = '% Generated from results.json SHA256: ' + hashlib.sha256(SOURCE.read_bytes()).hexdigest() + '\n'
    (OUT / 'numbers.tex').write_text(header + '\n'.join(
        r'\expandafter\def\csname result@' + k + r'\endcsname{' + fmt(v, k) + '}' for k, v in sorted(vals.items())) + '\n')
    (OUT / 'number_provenance.json').write_text(json.dumps(vals, indent=2) + '\n')
    missing = set(re.findall(r'\\Result\{([^{}]+)\}', (HERE/'paper.tex').read_text())) - vals.keys()
    if missing:
        raise ValueError(f'Missing result keys: {sorted(missing)}')

ENCODERS = {'mbert': 'mBERT', 'xlmr_base': 'XLM-R-base', 'labse': 'LaBSE'}

def tables():
    rows, nulls = [], []
    for name, d in R['domains'].items():
        for m, x in d['metrics'].items():
            row = [NAMES[name], 'E' if m == 'euclid' else 'C']
            row += [fmt(x[k]) for k in ('distance_r', 'distance_quartet', 'nn_purity', 'nn_chance')]
            row += [f"{x[t]['recovered']}/{x[t]['available']}" for t in ('nj', 'fastme')]
            row += [fmt(x[t]['tree_quartet']) for t in ('nj', 'fastme')]
            rows.append(row)
            p = x['permutation']
            nulls.append(row[:2] + [x['n_resolved'], x['n_quartets']] +
                [fmt(p[k], k) for k in ('distance_r_p', 'distance_quartet_p', 'nn_purity_p')] +
                [f"{x[k]:.4f}" for k in ('four_point_mean', 'four_point_sample_max')])
    table('domains', 'llrrrrrrrr', ['Domain', '', '$r$', '$Q_D$', 'NN', 'Null NN', 'NJ splits', 'FM splits', '$Q_{NJ}$', '$Q_{FM}$'], rows)
    table('nulls', 'llrrrrrrr', ['Domain', '', 'Resolved', 'Examined', '$p_r$', '$p_Q$', '$p_{NN}$', 'Mean residual', 'Max residual'], nulls)
    for pop in ('36', '89'):
        keys = ['raw', 'raw_centered', 'aligned89', 'aligned89_centered']
        names = ['Raw', 'Raw, centered', 'Align 89', 'Align 89, centered']
        if pop == '36':
            keys += ['aligned36', 'aligned36_centered']
            names += ['Align 36', 'Align 36, centered']
        rows = [[L, 'E' if m == 'euclid' else 'C'] + [fmt(layer[k][m]['r']) for k in keys]
                for L, layer in R['readouts']['language'][pop]['layers'].items() for m in COLORS]
        table('readouts'+pop, 'll'+'r'*len(keys), ['Layer', '']+names, rows)
    rows = []
    for L, d in R['readouts']['speech'].items():
        x = d['metrics']['cosine']
        rows.append([L, f"{d['participation_ratio']:.2f}", f"{d['mean_pairwise_cosine']:.4f}",
            fmt(x['distance_quartet']), fmt(x['nn_purity'])] +
            [f"{x[t]['recovered']}/{x[t]['available']}" for t in ('nj', 'fastme')] +
            [fmt(x[t]['tree_quartet']) for t in ('nj', 'fastme')])
    table('speech', 'rrrrrrrrr', ['Layer', 'PR', 'Mean cosine', '$Q_D$', 'NN', 'NJ splits', 'FM splits', '$Q_{NJ}$', '$Q_{FM}$'], rows)
    rows = []
    for k, name in [('imputed_all','Imputed, all pairs'), ('imputed_matched','Imputed, matched pairs'),
        ('observed_matched','Observed, matched pairs'), ('wals','WALS only'), ('reverse_observed','Reverse partial, observed')]:
        d = R['language_appendix']['descent'][k]
        rows.append([name, d['n_pairs'], fmt(d['r']), f"[{d['source_deletion_min']:.3f}, {d['source_deletion_max']:.3f}]"])
    table('controls', 'lrrr', ['Control and pair set', 'Pairs', 'Partial $r$', 'Source deletion range'], rows)
    rows = []
    for name, block in R['language_appendix']['clock'].items():
        a, b = block['pooled'], block['without_germanic']
        rows.append([name.capitalize(), fmt(a['r']), fmt(a['rho']), fmt(a['p_taxon'],'p_taxon'),
            f"[{a['source_deletion_min']:.3f}, {a['source_deletion_max']:.3f}]", fmt(b['r']), fmt(b['p_taxon'],'p_taxon')])
    table('clock', 'lrrrrrr', ['Dates', '$r$', r'$\rho$', '$p$', 'Deletion range', '$r$ w/o Germ.', '$p$ w/o Germ.'], rows)
    rows = []
    for name, d in R['language_appendix']['replication'].items():
        a, c = d['descent'], d['clock']['pooled']
        rows.append([ENCODERS.get(name, name.replace('_',r'\_'))] + [fmt(a[k]['r']) for k in ('imputed_all','imputed_matched','observed_matched')] +
                    [fmt(c['r']), fmt(c['p_taxon'],'p_taxon')])
    table('replication', 'lrrrrr', ['Encoder','Imputed 94','Imputed 81','Observed 81','Clock $r$','Clock $p$'], rows)
    rows = [[f"{c['radial']:g}", f"{c['mixing']:g}", 'E' if m=='euclid' else 'C'] +
        [fmt(float(syn(cell,m,*path).mean())) for path in PATHS]
        for cell,c in R['synthetic']['seeds'][0]['cells'].items() for m in COLORS]
    table('synthetic', 'rrrrrrrrrr', ['$k_N$','$k_C$','','$r$','$Q_D$','$S_{NJ}$','$S_{FM}$','$Q_{NJ}$','$Q_{FM}$','Deep $S_{NJ}$'], rows)
    table('clouds', 'lrr', ['Statistic','Layer','Partial $r$'],
        [[x['channel'],x['layer'],fmt(x['r'])] for x in R['archive']['cloud_shape']])

def save(fig, name):
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / name)
    plt.close(fig)

def crossmodal():
    fig, axes = plt.subplots(1,3,figsize=(7.4,2.9),sharey=True)
    for i,(name,d) in enumerate(R['domains'].items()):
        for m,offset in [('euclid',-.13),('cosine',.13)]:
            x=d['metrics'][m]
            axes[0].plot(x['distance_quartet'],i+offset,'o',color=COLORS[m],ms=6)
            for method,marker,extra in [('nj','o',-.04),('fastme','s',.04)]:
                for ax,key in [(axes[1],'tree_quartet'),(axes[2],'split_recall')]:
                    ax.plot(x[method][key],i+offset+extra,marker,color=COLORS[m],ms=5,
                            mfc=COLORS[m] if method=='nj' else 'white')
    axes[0].set_yticks(range(5),list(NAMES.values()))
    axes[0].invert_yaxis()
    for ax,title in zip(axes,['(a) Distance\nquartets','(b) Reconstructed-tree\nquartets','(c) Reference-split\nrecall']):
        ax.set_title(title,fontsize=10); ax.set_xlim(-.02,1.02)
        ax.set_xticks([0,1/3,2/3,1],['0','1/3','2/3','1']); ax.grid(axis='x',alpha=.18)
    for ax in axes[:2]: ax.axvline(1/3,ls=':',color='.55')
    for m in COLORS: axes[0].plot([],[],'o',color=COLORS[m],label=m.capitalize())
    for name,marker in [('NJ','o'),('FastME','s')]:
        axes[1].plot([],[],marker,color='.3',mfc='.3' if name=='NJ' else 'white',label=name)
    for ax in axes[:2]: ax.legend(loc='upper left',bbox_to_anchor=(0,-.16),ncol=2,frameon=False)
    save(fig,'fig1_recovery.pdf')

def readouts():
    fig,ax=plt.subplots(1,3,figsize=(7.4,3.0))
    L=R['readouts']['language']['36']['layers']['12']
    for m,shift in [('euclid',-.1),('cosine',.1)]:
        ds=[L[k][m] for k in ['raw','raw_centered','aligned36']]
        y=np.array([d['r'] for d in ds]); lo=np.array([d['source_deletion_min'] for d in ds]); hi=np.array([d['source_deletion_max'] for d in ds])
        ax[0].errorbar(np.arange(3)+shift,y,yerr=[y-lo,hi-y],fmt='o',capsize=2,color=COLORS[m],label=m.capitalize())
    ax[0].set_xticks(range(3),['Raw','Raw,\ncentered','Aligned\n36']); ax[0].set_ylabel('Partial correlation'); ax[0].set_ylim(0,.51)
    ax[0].set_title('(a) Origin and\nnormalization',fontsize=10)
    for pop,color in [('36','#3974a4'),('89','#6f4e7c')]:
        ax[1].plot([0,1],[L['aligned'+pop+k]['cosine']['r'] for k in ('','_centered')],'o-',color=color,label=f'Fit on {pop}')
    ax[1].set_xticks([0,1],['Default\norigin','Evaluation\nmean origin']); ax[1].set_xlim(-.15,1.15); ax[1].set_ylim(.25,.5)
    ax[1].set_title('(b) Same 36\nlanguages',fontsize=10)
    for a in ax[:2]: a.legend(loc='upper left',bbox_to_anchor=(-.12,-.29),ncol=1,frameon=False)
    layers=list(R['readouts']['speech']); xx=[int(k) for k in layers]
    for m in COLORS:
        ax[2].plot(xx,[R['readouts']['speech'][k]['metrics'][m]['distance_quartet'] for k in layers],'o-',color=COLORS[m],ms=4)
    ax[2].axhline(1/3,color='.6',ls=':'); ax[2].set_ylim(.28,.66); ax[2].set_xticks(xx)
    ax[2].set_xlabel('Speech layer'); ax[2].set_ylabel('Quartet agreement'); ax[2].set_title('(c) Layer\nsensitivity',fontsize=10)
    right=ax[2].twinx(); right.plot(xx,[R['readouts']['speech'][k]['participation_ratio'] for k in layers],'s--',color='.4',ms=4)
    right.set_ylim(0,14); right.set_ylabel('PR (dashed)',color='.4')
    save(fig,'fig2_readouts.pdf')

def synthetic():
    fig,axes=plt.subplots(1,3,figsize=(7.4,3.0))
    radial=['clean','radial_0.5','radial_1.0','radial_1.5','radial_2.0']
    mixed=['clean','mixing_0.1','mixing_0.2','mixing_0.35','mixing_0.5']
    def curve(ax,cells,doses,m,path,color,label,ls='-'):
        a=np.stack([syn(c,m,*path) for c in cells]); y=a.mean(1); sd=a.std(1,ddof=1)
        ax.plot(doses,y,ls,color=color,label=label,marker='o',ms=3)
        ax.fill_between(doses,y-sd,y+sd,color=color,alpha=.10)
    for m in COLORS: curve(axes[0],radial,[0,.5,1,1.5,2],m,('distance_r',),COLORS[m],m.capitalize())
    for path,color,label,ls in [(('distance_r',),'#555555','$r$',':'),(('distance_quartet',),'#33846d','$Q_D$','-'),
        (('nj','tree_quartet'),'#3974a4','$Q_{NJ}$','-'),(('fastme','tree_quartet'),'#c06932','$Q_{FM}$','--')]:
        curve(axes[1],mixed,[0,.1,.2,.35,.5],'euclid',path,color,label,ls)
    for m in COLORS:
        for method,ls in [('nj','-'),('fastme','--')]:
            curve(axes[2],mixed,[0,.1,.2,.35,.5],m,(method,'split_recall'),COLORS[m],
                ('E' if m=='euclid' else 'C')+', '+('NJ' if method=='nj' else 'FM'),ls)
    for ax,title,xlabel in zip(axes,['(a) Radial scaling:\ncorrelation','(b) Mixing: graded\nagreement (E)','(c) Mixing: all\nreference splits'],
        ['Radial amplitude $k_N$','Mixing weight $k_C$','Mixing weight $k_C$']):
        ax.set_title(title,fontsize=10); ax.set_xlabel(xlabel); ax.set_ylim(-.06,1.04); ax.grid(alpha=.18)
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.23),ncol=2,frameon=False)
    save(fig,'fig3_perturbations.pdf')

if __name__ == '__main__':
    numbers(); tables(); crossmodal(); readouts(); synthetic()
    print('Generated numeric macros, provenance, tables, and three figures.')
