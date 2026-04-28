set terminal pngcairo size 800,600
set output outfile
set yrange [-1.2:1.2]
set xrange [-1.2:1.2]
plot infile index 1 using 1:2 title "Re[P(x)]" with lines linewidth 2 lc rgb "green", \
infile index 1 using 1:3 title "Im[P(x)]" with lines linewidth 2 lc rgb "red", \
infile index 0 using 1:2 title "Re[y-target]" with points pointtype 7 lc rgb "blue", \
infile index 0 using 1:3 title "Im[y-target]" with points pointtype 7 lc rgb "orange"
