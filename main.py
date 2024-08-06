import pyramids
from hand_tracking import tracking
import fastfouriertransform
import hrcalculator

freq_min = 1
freq_max = 1.8
video_frames, frame_length, fps = tracking()

print("Image Preprocessing...")
pvid = pyramids.build_video_pyramid(video_frames)

amplified_video_pyramid = []

for i, video in enumerate(pvid):
    if i == 0 or i == len(pvid) - 1:
        continue
    print("Running Fast Fourier Transform and Eulerian magnification...")
    result, fft, frequencies = fastfouriertransform.transform(video, freq_min, freq_max, fps)
    pvid[i] += result
    print("Calculating heart rate...")
    heart_rate = hrcalculator.hr(fft, frequencies, freq_min, freq_max)

# Output heart rate and final video
print("Heart rate: ", heart_rate, "bpm")
