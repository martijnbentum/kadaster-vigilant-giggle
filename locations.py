import glob

base = '../'
scans = base + 'kadaster_scans/'
with_stamp = scans + 'with_stamp/'
no_stamp = scans + 'no_stamp/'
with_note = scans + 'with_note/'
no_note = scans + 'no_note/'

pickle = base + 'kadaster_pickle/'

no_stamp_files = glob.glob(no_stamp + '*.jpg')
with_stamp_files = glob.glob(with_stamp + '*.jpg')
no_note_files = glob.glob(no_note + '*.jpg')
with_note_files = glob.glob(with_note + '*.jpg')



