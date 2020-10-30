set terminal postscript eps enhanced color font "Times-Roman,40"
set output 'operator_belief_error_op.eps'
set style fill  transparent solid 0.5 noborder
set style circle radius 0.15
set yrange [0:21]
set xrange [1:200]
set xtics ("0" 1, "150" 30, "350" 70, "500" 100, "1000" 200) 
set ytics ("0" 0, "0.5" 10.5, "1" 21)
set xlabel "Simulation steps"
set ylabel "Error in belief map" offset 1,0,0
set datafile separator ","
set object 1 rect fc rgb "#88cc88" from 30,graph 0 to 70,graph 1 fillstyle solid 0.4 noborder 

plot 'operator_belief_error_combined_percentiles.csv' using 1:2:4  w filledcu lt rgb "#f79100" notitle,\
'operator_belief_error_combined_percentiles_op.csv' using 1:2:4  w filledcu lt rgb "#683d02" notitle,\
'operator_belief_error_combined_percentiles.csv' using 1:3 w lines lt rgb "#FF0000" notitle,\
'operator_belief_error_combined_percentiles_op.csv' using 1:3 w lines lt rgb "#0000FF" notitle,
