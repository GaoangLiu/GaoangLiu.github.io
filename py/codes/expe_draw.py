import matplotlib.pyplot as plt
import numpy as np 


datafile = "./data/vnstat-d.dat"
days = []
rx = [] 
tx = [] 
total = []
with open(datafile, 'r') as f:
    for line in f:
        line = line.rstrip().lstrip()
        if len(line) and line[0].isdigit():
            d, r, _, _, t, _, _, tt = line.split()[0:8]
            # print(d, r, t, tt)
            d = '.'.join(d.split("/")[0:2])
            days.append(d)
            rx.append(float(r))
            tx.append(float(t))
            total.append(tt)

for style in ['seaborn-darkgrid'] :#plt.style.available:
	print(style)
	plt.rcParams.update({'figure.autolayout': True})
	# plt.style.use('grayscale')
	plt.style.use(style)
	fig, ax = plt.subplots(figsize=(6, 8))
	ax.barh(days, rx, height=0.2)
	labels = ax.get_xticklabels()
	plt.setp(labels, rotation=45, horizontalalignment='right')
	ax.set(xlabel='Bandwidth (GiB)', ylabel='Date',
       title='Network Overview')
	plt.show(block=True)
	# plt.pause(3)
	# plt.close()
# fig.savefig("vnstat.svg", transparent=False, dpi=80)