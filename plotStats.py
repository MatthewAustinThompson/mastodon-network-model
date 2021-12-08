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
	if( len(sys.argv) != 3 ):
		sys.stderr.write("USAGE: %s <data.csv> <output.png>\n" % sys.argv[0])
		sys.exit(1)
	infile = sys.argv[1]
	outfile = sys.argv[2]
	df = pd.read_csv(infile)
	print("Verification that the CSV file parsed correctly:")
	print(df)
	df["index"] = df.index
	df = df.melt(id_vars=["index", "p", "exponent", "nodes_per_instance"], value_vars=["instance_connectivity","simpsons_index","reachability"])
	print("After melting to put in a convenient FacetGrid format:")
	print(df)
	g = sns.FacetGrid(data=df, col="variable", row="exponent", hue="nodes_per_instance", sharex=True, sharey=True)
	g.map(sns.scatterplot, "p", "value", alpha=0.3)
	g.add_legend(loc="lower center", bbox_to_anchor=(0.45, -0.06))
	g.set_titles(col_template="{col_name}")
	plt.savefig(outfile, bbox_inches="tight")
