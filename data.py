import pretty_midi
import torch
import numpy as np

import argparse
import logging
import pickle

logging.basicConfig(level=logging.INFO)

def parse_midi_file(file_path):
    # MIDI 파일 읽기
    midi_data = pretty_midi.PrettyMIDI(file_path)
    
    # 시간 단계당 음악 이벤트 추출
    events = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            events.append((note.start, note.end, note.pitch, instrument.program))
    
    # 이벤트 정렬
    events.sort(key=lambda x: x[0])
    
    return events

def preprocess_midi_events(events, seq_length):
    # MIDI 이벤트를 벡터로 변환
    event_vectors = []
    for event in events:
        event_vector = np.zeros((128 + 128 + 1))  # 128 for pitch, 128 for instrument program, 1 for event type
        event_vector[event[2]] = 1  # Pitch
        event_vector[128 + event[3]] = 1  # Instrument program
        event_vector[-1] = 1  # Event type
        event_vectors.append(event_vector)
    
    # 데이터 패딩
    while len(event_vectors) < seq_length:
        event_vectors.append(np.zeros_like(event_vectors[0]))
    
    # 시퀀스 자르기
    event_vectors = event_vectors[:seq_length]
    
    # 텐서로 변환
    event_tensor = torch.tensor(event_vectors, dtype=torch.float32)
    
    return event_tensor

def save_preprocessed_data(midi_tensor, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(midi_tensor, f)


def main(args):
    # MIDI 파일을 읽고 데이터로 변환
    logging.info(f"Reading MIDI file: {args.input_file}")
    events = parse_midi_file(args.input_file)
    seq_length = args.seq_length
    logging.info("Preprocessing MIDI events")
    midi_tensor = preprocess_midi_events(events, seq_length)

    # 전처리된 데이터 저장
    logging.info(f"Saving preprocessed data to: {args.output_file}")
    save_preprocessed_data(midi_tensor, args.output_file)
    logging.info("Preprocessing complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI data")
    parser.add_argument("--input_file", type=str, required=True, help="Input MIDI file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output preprocessed data file path")
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length for preprocessing")
    args = parser.parse_args()

    main(args)
