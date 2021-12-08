#!/usr/bin/env python3
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This script plots our three measured attributes:
#   - connectivity
#   - simpsons diversity index
#   - reachability
# Against our two sweeped parameters:
#   - p, the small-world rewiring probability
#   - the exponent for the power-law preferential attachment

if __name__ == "__main__":
	if( len(sys.argv) != 4 ):
		sys.stderr.write("USAGE: %s <data.csv> <output_linear.png> <output_log.png>\n" % sys.argv[0])
		sys.exit(1)
	infile = sys.argv[1]
	outfile_linear = sys.argv[2]
	outfile_log = sys.argv[3]
	df = pd.read_csv(infile)
	print("Verification that the CSV file parsed correctly:")
	print(df)
	df["index"] = df.index
	df = df.melt(id_vars=["index", "p", "exponent", "nodes_per_instance"], value_vars=["instance_connectivity","simpsons_index","reachability"])
	print("After melting to put in a convenient FacetGrid format:")
	print(df)

	# Now that data is parsed, plot in linear space
	g = sns.FacetGrid(data=df, col="variable", row="exponent", hue="nodes_per_instance", sharex=True, sharey=True)
	g.map(sns.scatterplot, "p", "value", alpha=0.3)
	g.add_legend(loc="lower center", bbox_to_anchor=(0.45, -0.06))
	g.set_titles(col_template="{col_name}")
	plt.savefig(outfile_linear, bbox_inches="tight")
	plt.clf()

	# Plot again in logspace, this time with *different* y-axis, because
	# different attributes have different lower bounds
	g = sns.FacetGrid(data=df, col="variable", row="exponent", hue="nodes_per_instance", sharex=True, sharey=False)
	g.map(sns.scatterplot, "p", "value", alpha=0.3)
	g.add_legend(loc="lower center", bbox_to_anchor=(0.45, -0.06))
	g.set_titles(col_template="{col_name}")
	g.set(xscale="log", xlim=(0.0001,1))
	g.set(yscale="log")
	axes = g.axes
	for row in range(0,3):
		axes[row,0].set_ylim(0.08,1)
		axes[row,1].set_ylim(0.0001,1)
		axes[row,2].set_ylim(0.05,1)
	#g.set(yscale="log", ylim=(0.0001,1))
	plt.savefig(outfile_log, bbox_inches="tight")
