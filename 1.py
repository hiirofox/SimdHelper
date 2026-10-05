import re
import matplotlib.pyplot as plt

raw = """1 ch: scalar=30382200 simd=35935300 speedup=0.84547X
2 ch: scalar=62547700 simd=52200800 speedup=1.19821X
3 ch: scalar=101775600 simd=91866500 speedup=1.10786X
4 ch: scalar=130391400 simd=101210300 speedup=1.28832X
5 ch: scalar=158272300 simd=133338800 speedup=1.18699X
6 ch: scalar=187030700 simd=154062800 speedup=1.21399X
7 ch: scalar=213928700 simd=171528900 speedup=1.24719X
8 ch: scalar=243688200 simd=213842500 speedup=1.13957X
9 ch: scalar=275953700 simd=232173400 speedup=1.18857X
10 ch: scalar=298415300 simd=250850400 speedup=1.18961X
11 ch: scalar=330702200 simd=271997700 speedup=1.21583X
12 ch: scalar=359087700 simd=201550000 speedup=1.78163X
13 ch: scalar=392112600 simd=297253000 speedup=1.31912X
14 ch: scalar=420785900 simd=339652300 speedup=1.23887X
15 ch: scalar=459167200 simd=399697700 speedup=1.14879X
16 ch: scalar=483540200 simd=278874500 speedup=1.73390X
17 ch: scalar=505857800 simd=350915000 speedup=1.44154X
18 ch: scalar=537108400 simd=379849600 speedup=1.41400X
19 ch: scalar=573283500 simd=424473300 speedup=1.35058X
20 ch: scalar=611974900 simd=341291700 speedup=1.79311X
21 ch: scalar=657252500 simd=406889400 speedup=1.61531X
22 ch: scalar=667775700 simd=443919300 speedup=1.50427X
23 ch: scalar=699033800 simd=472115800 speedup=1.48064X
24 ch: scalar=736938200 simd=385610900 speedup=1.91109X
25 ch: scalar=751866700 simd=426033100 speedup=1.76481X
26 ch: scalar=784211800 simd=450225600 speedup=1.74182X
27 ch: scalar=819134400 simd=509938600 speedup=1.60634X
28 ch: scalar=846185400 simd=630985900 speedup=1.34105X
29 ch: scalar=885287300 simd=682281300 speedup=1.29754X
30 ch: scalar=898963900 simd=698797100 speedup=1.28644X
31 ch: scalar=941154100 simd=736618100 speedup=1.27767X
32 ch: scalar=998653900 simd=537231600 speedup=1.85889X
33 ch: scalar=1029413800 simd=561888300 speedup=1.83206X
34 ch: scalar=1048417600 simd=588519900 speedup=1.78145X
35 ch: scalar=1085496300 simd=648222800 speedup=1.67457X
36 ch: scalar=1102174600 simd=648396100 speedup=1.69985X"""

pat = re.compile(r"(\d+)\s+ch:\s+scalar=(\d+)\s+simd=(\d+)\s+speedup=([\d.]+)X")
data = [tuple(map(float, m.groups())) for m in pat.finditer(raw)]
ch = [x[0] for x in data]
scalar = [x[1] for x in data]
simd = [x[2] for x in data]
speedup = [x[3] for x in data]
fig, ax1 = plt.subplots(figsize=(10, 5.8))

ax1.plot(ch, scalar, marker='o', markersize=3, linewidth=1.5, label='Scalar')
ax1.plot(ch, simd, marker='o', markersize=3, linewidth=1.5, label='SIMD')
ax1.set_xlabel('Channels')
ax1.set_ylabel('Clock ticks')
ax1.set_xlim(1, 36)
ax1.set_xticks(range(4, 37, 4))
ax1.grid(True, alpha=0.25)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(ch, speedup, color='green', marker='s', markersize=3, linewidth=1.5, linestyle='--', label='Speedup')
ax2.set_ylabel('Speedup (×)')
ax2.axhline(1.0, linewidth=0.8, linestyle=':')
ax2.legend(loc='upper right')

fig.tight_layout()
path = "granular_simd_scaling.png"
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.show()