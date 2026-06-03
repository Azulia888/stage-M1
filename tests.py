import exiftool
from pymediainfo import MediaInfo


info = MediaInfo.parse("videos/286.mp4")
for track in info.tracks:
    print(track.track_type, track.to_data())

print()
print("="*50)
print()

with exiftool.ExifToolHelper() as et:
    meta = et.get_metadata("videos/286.mp4")
    print(meta)