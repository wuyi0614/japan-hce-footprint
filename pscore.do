// Run propensity scores for japan hosuehold
//

clear
cd "/Users/mario/Documents/github/japan-hce-aging/data"
// import delimited "cache-scaled-data-11010137.csv", clear
import delimited "clustered-cache-11051552.csv", clear
set more off

gen pv1 = -(pv - 2)
gen lninc = ln(res_income_percap + 1)

// Set variables
local conts drive_freq car_num if_single_elderly dist_all ///
		heat_time if_single heater_4 build_space fridge_num heater_6 heat_room_num avg_age ///
		app_13 tv_time if_others ac_num double_window elderly_num heater_3  ///
		ener_saving_rate if_couple_kids build_type res_size lninc

local dummies i.prefecture i.city_class 

// Run OLS on emits_per
// {0: "6HFK", 1: "1UHFK", 2: "5LES", 3: "2UHBF", 4: "3MCK", 5: "4ULS"}
forvalues i=0/5{
	reg em_ex_car pv `conts' if cluster == `i'
	di "cluster-`i'"
}

// Before predictionis, create new variables 
// Predict probability for PV
logit pv `conts' `dummies'
predict pv_pscore
sum pv_pscore

// visualize 
qui summarize pv_pscore if pv & e(sample) 
local treat_total = r(N)
qui summarize pv_pscore if !pv & e(sample) 
local control_total = r(N)
di "Treated `treat_total'; Controls `control_total'"


// Predict probability for NEV
logit nev `conts' `dummies'
predict nev_pscore
sum nev_pscore

// visualize
qui summarize nev_pscore if nev & e(sample) 
local treat_total = r(N)
qui summarize nev_pscore if !nev & e(sample) 
local control_total = r(N)
di "Treated `treat_total'; Controls `control_total'"


// twoway (hist nev_pscore if pv==1 , width(0.01) fcolor(gs7) freq) ///
// 	 (hist nev_pscore if pv==0 , width(0.01) fcolor(none) lcolor(black) freq), ///
// 	legend(order(1 "Treated" 2 "Controls") size(small) ) scheme(sj) ///
// 	xtitle("Propensity Score", size(small)) xlabel(0.05(0.05)1, labsize(vsmall)) ///
// 	ytitle("Number of Stores", size(small)) ylabel(, labsize(vsmall)) ///
// 	note("Treated `treat_total'; Potential controls `control_total'")


// Output predicted data
export delimited using "cache-pscore-data.csv", replace
