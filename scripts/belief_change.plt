set terminal postscript eps enhanced color font "Times-Roman,34"
set output 'belief_error.eps'
set style fill  transparent solid 0.5 noborder
set style circle radius 0.15
set yrange [0:21]
set xrange [1:200]
set xtics ("0" 1, "150" 30, "350" 70, "500" 100, "1000" 200) 
set ytics ("0.5" 11, "1" 21)  
set xlabel "Simulation steps"
set ylabel "Error in belief map" offset 1.5,0,0
#set key samplen 2
#set size 1.05,1.01
set object 1 rect fc rgb "#88cc88" from 30,graph 0 to 70,graph 1 fillstyle solid 0.4 noborder 
set datafile separator ","

#set arrow from 23,0 to 23,160 nohead lt rgb "#0179ff" linewidth 5 dashtype "."
#plot 'stat_all.txt' using 1:2  every 10 with circles lc rgb "#88cc88" notitle,
#plot 'stat_all.txt' using 1:2  with line lw 3 lt 1 lc rgb "#FF0000" title "Right side",
#'stat_all.txt' using 1:3  with line lw 3 lt 2 lc rgb "#0000FF"  title "Left side"
plot 'belief_error_combined_percentiles.csv' using 1:2:4  w filledcu lt rgb "#f79100" notitle,\
'belief_error_combined_percentiles.csv' using 1:3 w lines lt rgb "#FF0000" notitle,
