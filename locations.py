import glob

base = '../'
scans = base + 'kadaster_scans/'
with_stamp = scans + 'with_stamp/'
no_stamp = scans + 'no_stamp/'
with_note = scans + 'with_note/'
no_note = scans + 'no_note/'

pickle = base + 'kadaster_pickle/'
cropped_name = base + 'kadaster_cropped_name/'
cropped_top = base + 'kadaster_cropped_top/'

viti_outputs= base + 'kadaster_viti_outputs/'

no_stamp_files = glob.glob(no_stamp + '*.jpg')
with_stamp_files = glob.glob(with_stamp + '*.jpg')
no_note_files = glob.glob(no_note + '*.jpg')
with_note_files = glob.glob(with_note + '*.jpg')

all_files = no_stamp_files + with_stamp_files + no_note_files + with_note_files




