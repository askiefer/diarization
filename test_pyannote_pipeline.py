import torch
# import torchaudio
import json
import time
import os
import pymysql
import csv
# from pyannote.audio import Pipeline


HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.0",
#     use_auth_token=HUGGING_FACE_API_TOKEN
# )
#
# pipeline.to(torch.device("mps"))


def run_pyannote(mp4_file: str):
    waveform, sample_rate = torchaudio.load(mp4_file)
    start = time.time()
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    end = time.time()
    total_runtime = (end - start) / 60
    print(f"Diarized with pyannote: {mp4_file}, took {total_runtime}m")
    return diarization

def write_diarization_tags(diarization, file_id):
    json_filepath = f"diarization/{file_id}.json"  # prepend json to the filepath for some reason
    speakers = set()
    utterances = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_tag = speaker.split("SPEAKER_")[1]
        speakers.add(speaker_tag)
        utterance = {
            "start": turn.start,
            "end": turn.end,
            "speaker": f"S{speaker_tag.zfill(2)}",
        }
        utterances.append(utterance)
    print(f"Found {len(speakers)} for {file_id}")
    data = {"utterances": utterances, "id": file_id}
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def get_transcript(video: str):
    mydb = os.getenv("DB_NAME")
    myhost = os.getenv("DB_HOST")
    myuser = os.getenv("DB_USER")
    mypass = os.getenv("DB_PASSWORD")

    connection = pymysql.connect(
        host=myhost, user=myuser, password=mypass, database=mydb
    )
    utterances = []
    sql = f"""
    select uid, vid, pid, diarizationTag, time, endTime, text from Utterance where vid={video}
    """
    with connection.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
        utterances = [r for r in result]
    return utterances


def run_pyannote():
    for s3_file in s3_files:
        diarization = run_pyannote(s3_file)
        file = os.path.basename(os.path.normpath(s3_file))
        file_id, ext = os.path.splitext(file)
        write_diarization_tags(diarization, file_id)


def merge_consecutive_utterances(utterances):
    if not utterances:
        return utterances

    merged_utterances = [utterances[0]]

    for i in range(1, len(utterances)):
        current_utterance = utterances[i]
        previous_utterance = merged_utterances[-1]

        if current_utterance['speaker'] == previous_utterance['speaker']:
            # Merge consecutive utterances with the same speaker
            previous_utterance['end'] = current_utterance['end']
        else:
            # Add the current utterance if the speaker is different
            merged_utterances.append(current_utterance)

    return merged_utterances

def compare_utterances_v2(s3_files):
    mapping_file_video_id = {
        "f6f9164c941a56540964a92b264b8b84": 44512,
        "9ebe812c5cbe9ef59dac40ef92ce2cc4": 44513,
        "bd3ded311505d363bdb52ce5ad699ba7": 44514,
        "2ed783406037113bd328803204f18dcb": 44515,
        "dd739dd99b5ccb9aeccea33f32a88eb3": 44516,

        "3c01fd9b76eb9559ac3bb0fd7c2e653c": 44636,
        "754d74e474a98b7cb27b8c9389f2663e": 44637,
        "2a85bf7205c8344b8e7a383c428802a5": 44638,
        "ea3ac11991759dc7613a9ec8f830abe6": 44639,
        "7699f5409f26ae4f247f54450559f927": 44640,
        "56b67c067a10a316fd8d5eb4eb8eb159": 44641,
        "24ea0fdd835e4a79c0171b95450214d9": 44642,
        "c6e1f16ecf7cfc7fee94d23b398fd43a": 44643,

        "80426c1ff2bc72089c4ea9b7746e991b": 45014,
        "a63b82845f26fb7b23ff903154024ef8": 45015,
        "273af2cce276610696990b0d79fe230f": 45016,
        "4c502b16df36e167a43a05d7e960305b": 45017,
        "a62894e232ac20ddb00c3fc476c01f7a": 45018,
        "793a453a99d5654c02411b316ba29f03": 45019,
        "7b8fd4c7ea64d3ee8dd4a3a525a77853": 45020,
        "c6484aee7d888aced41617cea46cf4e3": 45021,
    }

    for i, s3_file in enumerate(s3_files):
        file = os.path.basename(os.path.normpath(s3_file))
        file_id, ext = os.path.splitext(file)
        filepath = f"diarization/{file_id}.json"
        with open(filepath, 'r') as f:
            diarization_dict = json.load(f)
        # import pdb; pdb.set_trace()
        # diarization_utterances = diarization_dict["utterances"]
        diarization_utterances = merge_consecutive_utterances(diarization_dict["utterances"])
        diarization_utterances = [du for du in diarization_utterances if du["end"] - du["start"] > 1]

        # with open(f'diarization/merged_{file_id}.csv', 'w') as f:
        #     writer = csv.DictWriter(f, fieldnames=['start', 'end', 'speaker'])
        #     writer.writeheader()
        #     writer.writerows(diarization_utterances)

        video_id = mapping_file_video_id[file_id]
        utterances = get_transcript(video_id)
        # utterances = get_transcript(mapping_video_ids[video_id])
        utterances = [u for u in utterances if u["endTime"] - u['time'] > 1]

        count_success_and_failure_v2(utterances, diarization_utterances, s3_file, video_id, i+1)
    return utterances, diarization_utterances


def count_success_and_failure(utterances, diarization_utterances):
    i, j = 0, 0
    utterance_starttimes = set([])
    successes = 0
    failures = 0
    mapping = {}
    merged = []
    while i < len(utterances) and j < len(diarization_utterances):
        # if i > len(utterances):
        #     break  # we are done, break out of the loop

        start = utterances[i]["time"]
        end = utterances[i]["endTime"]
        pid = utterances[i]["pid"]
        # text = utterances[i]["text"]

        d_start = round(diarization_utterances[j]["start"])
        d_end = round(diarization_utterances[j]["end"])
        d_tag = diarization_utterances[j]["speaker"]

        if (
            (d_start <= start and d_end <= end)
            or (d_start <= start and end <= d_end)
            or (start <= d_start and d_end <= end)
            or (start <= d_start and end <= d_end)
        ):
            previous_d_tags = mapping.get(pid, [])
            if not previous_d_tags or (previous_d_tags and d_tag in previous_d_tags):
                successes += 1
            else:
                failures += 1
            previous_d_tags.append(d_tag)
            mapping[pid] = previous_d_tags

        elif d_end < start:
            j += 1
        elif end < d_start:
            i += 1

    print(successes, failures)
    print(len(diarization_utterances), len(utterances))
    for pid, d_tags in mapping.items():
        print(pid, d_tags)

def count_success_and_failure_v2(utterances, diarization_utterances, file_id, video_id, num_video):

    successes, failures = 0, 0
    mapping = {}
    # import pdb; pdb.set_trace()
    for i in range(len(utterances)):

        start = utterances[i]["time"]
        end = utterances[i]["endTime"]
        pid = utterances[i]["pid"]
        
        j = 0
        max_overlapping_time = 0
        diarization_utterance_match = None

        # determines the match
        while j < len(diarization_utterances):
            d_start = round(diarization_utterances[j]["start"])
            d_end = round(diarization_utterances[j]["end"])
            # d_tag = diarization_utterances[j]["speaker"]

            if (
                (d_start <= start and d_end <= end)
                or (d_start <= start and end <= d_end)
                or (start <= d_start and d_end <= end)
                or (start <= d_start and end <= d_end)
            ):
                min_time = max(d_start, start)
                max_time = min(d_end, end)
                overlap = max_time - min_time
                if overlap > max_overlapping_time:
                    max_overlapping_time = overlap
                    diarization_utterance_match = diarization_utterances[j]
            j += 1

        # print("optimal match", start, end, diarization_utterance_match)
        if diarization_utterance_match:
            match_dia_tag = diarization_utterance_match["speaker"]

            # now that we've determined the optimal match, see if any previous mappings
            previous_d_tag = mapping.get(pid, None)
            if previous_d_tag:
                mapping_d_tag_pid = {v: k for k, v in mapping.items()}
                if match_dia_tag == previous_d_tag:
                    if match_dia_tag not in mapping_d_tag_pid:
                        successes += 1
                    elif mapping_d_tag_pid[match_dia_tag] == pid:
                        successes += 1
                    else:
                        failures += 1
                else:
                    failures += 1

            else:
                mapping[pid] = match_dia_tag
                successes += 1
        else:
            failures += 1

    print(num_video)
    print(f"file_id: {file_id}, video_id: {video_id}")
    print(f"successes: {successes}, failures: {failures}")
    print("utterance_length", len(utterances), "accuracy:", successes / len(utterances), "\n")


if __name__ == "__main__":
    s3_files = [
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/bd3ded311505d363bdb52ce5ad699ba7/bd3ded311505d363bdb52ce5ad699ba7.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/f6f9164c941a56540964a92b264b8b84/f6f9164c941a56540964a92b264b8b84.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/9ebe812c5cbe9ef59dac40ef92ce2cc4/9ebe812c5cbe9ef59dac40ef92ce2cc4.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/2ed783406037113bd328803204f18dcb/2ed783406037113bd328803204f18dcb.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/dd739dd99b5ccb9aeccea33f32a88eb3/dd739dd99b5ccb9aeccea33f32a88eb3.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/3c01fd9b76eb9559ac3bb0fd7c2e653c/3c01fd9b76eb9559ac3bb0fd7c2e653c.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/754d74e474a98b7cb27b8c9389f2663e/754d74e474a98b7cb27b8c9389f2663e.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/2a85bf7205c8344b8e7a383c428802a5/2a85bf7205c8344b8e7a383c428802a5.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/ea3ac11991759dc7613a9ec8f830abe6/ea3ac11991759dc7613a9ec8f830abe6.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/7699f5409f26ae4f247f54450559f927/7699f5409f26ae4f247f54450559f927.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/56b67c067a10a316fd8d5eb4eb8eb159/56b67c067a10a316fd8d5eb4eb8eb159.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/24ea0fdd835e4a79c0171b95450214d9/24ea0fdd835e4a79c0171b95450214d9.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/c6e1f16ecf7cfc7fee94d23b398fd43a/c6e1f16ecf7cfc7fee94d23b398fd43a.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/80426c1ff2bc72089c4ea9b7746e991b/80426c1ff2bc72089c4ea9b7746e991b.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/a63b82845f26fb7b23ff903154024ef8/a63b82845f26fb7b23ff903154024ef8.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/273af2cce276610696990b0d79fe230f/273af2cce276610696990b0d79fe230f.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/4c502b16df36e167a43a05d7e960305b/4c502b16df36e167a43a05d7e960305b.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/a62894e232ac20ddb00c3fc476c01f7a/a62894e232ac20ddb00c3fc476c01f7a.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/793a453a99d5654c02411b316ba29f03/793a453a99d5654c02411b316ba29f03.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/7b8fd4c7ea64d3ee8dd4a3a525a77853/7b8fd4c7ea64d3ee8dd4a3a525a77853.mp4",
        "https://s3-us-west-2.amazonaws.com/videostorage-us-west/videos/c6484aee7d888aced41617cea46cf4e3/c6484aee7d888aced41617cea46cf4e3.mp4",
    ]

    compare_utterances_v2(s3_files)
