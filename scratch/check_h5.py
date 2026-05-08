import h5py
files = ['logs.h5', 'logs_256.h5', 'logs_256_clean.h5']
for f in files:
    try:
        h5py.File(f, 'r')
        print(f + ' is OK')
    except Exception as e:
        print(f + ' FAILED: ' + str(e))
