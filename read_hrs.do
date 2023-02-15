

clear
set more off
foreach data in PX08A_R PX10A_R PX12A_R PX14A_R PX16A_R {
	global path "/Users/yiang/Downloads/Disposition_HRS/"
	infile using "$path/`data'.dct", using ("$path/`data'.da")
	save "$path/`data'.dta", replace
	clear
}
