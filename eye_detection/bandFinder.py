val1, val2, upper, lower = 0, 0, 100, 0

while True:
	print(upper, lower)
	val1 = raw_input("Upper: ")
	val2 = raw_input("Lower: ")
	if float(val1) == 0:
		break
	if float(val1) < float(upper):
		upper = val1
	if float(val2) > float(lower):
		lower = val2 
