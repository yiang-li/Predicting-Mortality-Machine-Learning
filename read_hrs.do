clear
set more off
cd "/Users/yiang/Downloads/Disposition_HRS/csv"
foreach data in px04A_R PX06A_R PX08A_R PX10A_R PX12A_R PX14A_R PX16A_R {
	global path "/Users/yiang/Downloads/Disposition_HRS"
	infile using "$path/`data'.dct", using ("$path/`data'.da")
	save "$path/data/`data'.dta", replace
	outsheet using `data'.csv , comma nolabel replace
	clear
}
clear
use "/Users/yiang/Downloads/Disposition_HRS/data/PX18A_R.dta", clear
outsheet using PX18A_R.csv , comma nolabel replace
clear


set more off
cd "/Users/yiang/Downloads/Core_HRS_2002/csv"
foreach data in H02C_R H02Q_H{
	global path "/Users/yiang/Downloads/Core_HRS_2002"
	infile using "$path/h02sta/`data'.dct", using ("$path/h02da/`data'.da")
	save "$path/data/`data'.dta", replace
	outsheet using `data'.csv , comma nolabel replace
	clear
}
