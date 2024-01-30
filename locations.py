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
kraken_outputs = base + 'kadaster_kraken/'
kraken_no_stamp = kraken_outputs + 'no_stamp/'
kraken_with_stamp = kraken_outputs + 'with_stamp/'
kraken_no_note = kraken_outputs + 'no_note/'
kraken_with_note = kraken_outputs + 'with_note/'

no_stamp_files = glob.glob(no_stamp + '*.jpg')
with_stamp_files = glob.glob(with_stamp + '*.jpg')
no_note_files = glob.glob(no_note + '*.jpg')
with_note_files = glob.glob(with_note + '*.jpg')

all_files = no_stamp_files + with_stamp_files + no_note_files + with_note_files


kraken_no_stamp_files = glob.glob(kraken_no_stamp + '*.json')
kraken_with_stamp_files = glob.glob(kraken_with_stamp + '*.json')
kraken_no_note_files = glob.glob(kraken_no_note + '*.json')
kraken_with_note_files = glob.glob(kraken_with_note + '*.json')

all_kraken_files =  kraken_with_note_files + kraken_no_note_files
all_kraken_files +=  kraken_with_stamp_files + kraken_no_stamp_files


# haarlem = base + 'haarlem/'
haarlem_base = '/Volumes/Extreme SSD/'
haarlem = haarlem_base + 'haarlem_input/'
haarlem_split_pages = haarlem_base + 'split_pages/'
haarlem_top_images = haarlem_base + 'top_images/'
haarlem_viti_outputs = haarlem_base + 'viti_outputs/'
