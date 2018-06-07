import os, sys

if len(sys.argv) != 2:
    raise Exception('')

cmnd = 'rm %s/Data/%s*' % (os.environ['LION_PATH'], sys.argv[1])
os.system(cmnd)
print cmnd
