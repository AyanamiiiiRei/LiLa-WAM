"""swap_blocks failure classes from the recorded trajectories (stage-memory experiment).

  python analysis/swap_blocks_progress.py [runs_dir]

Moves are read from gripper close/open positions (tray L/M/R, or the button). With
exp = [(A,C),(B,A),(C,B)] derived from the first move: press_after_k = pressed after k correct
moves; restart_after_3 = all 3 moves done, then started moving again; wrong_move_after_k = an
unexpected move; stuck_after_k = repeated the last move / never pressed (placement retries).
"""
import json,numpy as np,collections,sys
import pathlib
R=str(pathlib.Path(sys.argv[1] if len(sys.argv)>1 else 'runs'))+'/'
G={'A':'sm_A-seed0/rmbench_eval/ckpt_8000','F':'sm_F-seed0/rmbench_eval/ckpt_8000','M':'sm_M-seed0/rmbench_eval/ckpt_8000','M-empty':'sm_M-seed0/rmbench_eval/ckpt_8000_M-empty','M0':'sm_M0-seed0/rmbench_eval/ckpt_8000','M-proprio':'sm_M-seed0/rmbench_eval/ckpt_8000_M-proprio','M-head':'sm_M-seed0/rmbench_eval/ckpt_8000_M-head','M-time':'sm_M-seed0/rmbench_eval/ckpt_8000_M-time','S':'sm_S-seed0/rmbench_eval/ckpt_1500'}
def tray(x): return 'LMR'[int(np.clip(round(x/0.14),-1,1))+1]
def classify(st):
    ev=[]
    for o,gi,arm in ((0,7,'l'),(8,15,'r')):
        g=st[:,gi]<0.5
        for t in range(1,len(g)):
            if g[t]!=g[t-1]:
                btn=st[t,o+1]<-0.24
                ev.append((t,arm,'G' if g[t] else 'R','B' if btn else tray(st[t,o])))
    ev.sort()
    open_g={}; acts=[]
    for t,arm,k,p in ev:
        if k=='G':
            if p=='B': acts.append(('press',None)); open_g.pop(arm,None)
            else: open_g[arm]=p
        elif arm in open_g:
            s=open_g.pop(arm)
            if s!=p: acts.append(('move',(s,p)))
    moves=[a[1] for a in acts if a[0]=='move']
    if not moves: return 'no move',0
    A,C=moves[0]; B=({'L','M','R'}-{A,C}).pop(); exp=[(A,C),(B,A),(C,B)]
    idx=0; first_bad=None
    for kind,m in acts:
        if kind=='press':
            return ('press_after_%d'%idx if idx<3 else 'pressed_after_3'), idx
        if idx<3 and m==exp[idx]: idx+=1
        elif idx>0 and m==exp[idx-1]: continue        # redo of the previous move (placement retry)
        else: return ('restart_after_3' if idx==3 else 'wrong_move_after_%d'%idx), idx
    return ('stuck_after_%d'%idx), idx
out={}
import os
for g,p in G.items():
    if not os.path.isfile(R+p+'/episodes.jsonl'): print(g,'(no results yet)'); continue
    E={}
    for l in open(R+p+'/episodes.jsonl'):
        e=json.loads(l); E[e['seed']]=e
    c=collections.Counter(); per={}
    for s,e in sorted(E.items()):
        st=np.load(R+p+f'/policy_view/swap_blocks_seed{s}.npz')['states'][:e['env_steps']]
        lab,_=classify(st)
        if e['success']: lab='SUCCESS'
        per[s]=(lab,e.get('stages_done'))
        c[lab]+=1
    out[g]=per
    print(g, dict(sorted(c.items())))

