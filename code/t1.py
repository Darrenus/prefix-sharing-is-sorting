from core import *
# hand-built gadget: x=0 y=1 a=2 b=3 c=4
reqs = [frozenset({0,2,3}), frozenset({0,2,4}), frozenset({1,2,3}), frozenset({1,3,4})]
U = {0,1,2,3,4}
f,fa = opt_free(reqs)
g,ga = opt_global(reqs,U)
names = {0:'x',1:'y',2:'a',3:'b',4:'c'}
sh = lambda s: ''.join(names[z] for z in s)
print("requests:", [sorted(sh(sorted(r))) and sh(sorted(r)) for r in reqs])
print("FREE  opt =", f, " -> ", [sh(s) for s in fa])
print("GLOBAL opt =", g, " order:", sh(ga[0]), " -> ", [sh(s) for s in ga[1]])
