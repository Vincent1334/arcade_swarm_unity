set terminal postscript eps enhanced color font "Times-Roman,40"
set output 'operator_confidence.eps'
set style fill  transparent solid 0.5 noborder
set style circle radius 0.15
set yrange [0:1600]
set xrange [1:200]
set xtics ("0" 0, "500" 100, "1000" 200) 
set ytics 0, 800, 1600 
set xlabel "Simulation steps"
set ylabel "Confidence" offset 2,0,0
set datafile separator ","


plot 'operator_confidence_combined_percentiles.csv' using 1:2:4  w filledcu lt rgb "#683d02" notitle,\
'operator_confidence_combined_percentiles.csv' using 1:3 w lines lt rgb "#0000FF" notitle,
