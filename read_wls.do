use "/Users/yiang/Downloads/wls_bl_14.01.stata/wls_bl_14_01.dta", clear
ds , has(varlabel *RA* *R1* *R2* *R3* *R4* *R6* *R7*)
drop `r(varlist)'
*phone interview
drop if z_ga016rey == .
*mail interview
drop if z_ia004yre == .
outsheet using wls.csv , comma nolabel replace
