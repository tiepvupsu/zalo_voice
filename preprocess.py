from __future__ import print_function 
import os, subprocess
import config as cf

def convert_rate(src_folder, dst_folder):
    if not os.path.isdir(src_folder):
        print("Source folder doesn't exist, continue")
    if not os.path.isdir(dst_folder): os.makedirs(dst_folder)

    fns = [fn for fn in os.listdir(src_folder) if 
                any(map(fn.endswith, ['.mp3', '.wav', '.amr']))]

    for i, fn in enumerate(fns): 
        old_fn = os.path.join(src_folder, fn) 
        new_fn = os.path.join(dst_folder, fn + '.wav')
        # new_fn = dst_folder + fn + '.wav'
        if os.path.isfile(new_fn): continue
        # convert all file to wav, mono, sample rate 8000
        subprocess.call(['ffmpeg', '-loglevel', 'panic', '-i',  old_fn, 
                '-acodec', 'pcm_s16le', '-ac', '1', '-ar', cf.RATE, new_fn])
        if (i+1)%100 == 0:
            print('{}/{}: {}'.format(i+1, len(fns), new_fn))


if __name__ == '__main__':
    # train 
    for cate in cf.CATES:
        src_folder = os.path.join(cf.BASE_ORIGINAL_TRAIN, cate)
        dst_folder = os.path.join(cf.BASE_TRAIN, cate)
        convert_rate(src_folder, dst_folder)

    # public_test
    convert_rate(cf.BASE_ORIGINAL_PUBLIC_TEST, cf.BASE_PUBLIC_TEST)
    # private test 
    convert_rate(cf.BASE_ORIGINAL_PRIVATE_TEST, cf.BASE_PRIVATE_TEST)


