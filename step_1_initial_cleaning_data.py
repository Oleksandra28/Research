__author__ = 'osopova'


# import subprocess
# print "start"
# subprocess.call(['sh', "./test.sh"])
# print "end"

clean_data = True

###########################################################
### CLEAN DATA
###########################################################
if clean_data:
    import subprocess
    print "start"
    subprocess.call(['bash', "./preprocessing/awk_cleanall.bash"])
    print "end"



