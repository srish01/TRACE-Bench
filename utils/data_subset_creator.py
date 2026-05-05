# Python file to create subset of the Jsonl data for the FSS evaluation.

import json
import shutil
import os
from pathlib import Path
import argparse


def filter_jsonl_by_speaker(input_dir, output_dir, speaker_id="p227", max_entries=None, max_files=None, is_multi_turn=False):
    """
    Read all JSONL files from input_dir, filter entries by speaker_id,
    and save filtered results to output_dir.
    
    Args:
        input_dir (str): Path to directory containing JSONL files
        output_dir (str): Path to directory where filtered files will be saved
        speaker_id (str): Speaker ID to filter by (default: "p227"). Use "all" for all speakers.
        max_entries (int): Maximum number of entries per file (None or 0 for all)
        max_files (int): Maximum number of files to process (None or 0 for all)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    manifests_input_dir = os.path.join(input_dir, 'manifests/styletts2/')
    manifests_output_dir = os.path.join(output_dir, 'manifests')
    references_output_dir = os.path.join(output_dir, 'references', 'VCTK', f'{speaker_id}')
    styletts2_output_dir = os.path.join(output_dir, 'styletts2')

    Path(manifests_output_dir).mkdir(parents=True, exist_ok=True)
    Path(references_output_dir).mkdir(parents=True, exist_ok=True)
    Path(styletts2_output_dir).mkdir(parents=True, exist_ok=True)

    # Get all JSONL files in the input directory
    input_path = Path(manifests_input_dir)
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {manifests_input_dir}")
        return
    
    # Limit the number of files if max_files is specified
    if max_files and max_files > 0:
        jsonl_files = jsonl_files[:max_files]
    
    print(f"Found {len(jsonl_files)} JSONL file(s) to process")
    
    # Process each JSONL file
    total_entries = 0
    total_filtered = 0
    total_audio_copied = 0
    total_audio_failed = 0
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.name}")
        
        filtered_entries = []
        entry_count = 0
        file_filtered_count = 0
        
        # Read and filter entries
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        entry_count += 1
                        
                        # Check if speaker_id matches (or if 'all' is specified, match everything)
                        current_speaker = entry.get('reference', {}).get('speaker_id')
                        if speaker_id == 'all' or current_speaker == speaker_id:
                            # Check if we've reached the maximum number of entries for this file
                            if max_entries and max_entries > 0 and file_filtered_count >= max_entries:
                                continue
                            
                            filtered_entries.append(entry)
                            file_filtered_count += 1
                            
                            if is_multi_turn == False:
                                # Copy the generated audio file if it exists
                                generated = entry.get('generated', {})
                                if generated.get('ok') and generated.get('path'):
                                    audio_path = generated['path']
                                    source_audio = os.path.join(input_dir, audio_path)
                                    dest_audio = os.path.join(output_dir, audio_path)
                                    
                                    # Create destination directory if it doesn't exist
                                    dest_dir = os.path.dirname(dest_audio)
                                    Path(dest_dir).mkdir(parents=True, exist_ok=True)
                                    
                                    # Copy the audio file
                                    try:
                                        if os.path.exists(source_audio):
                                            shutil.copy2(source_audio, dest_audio)
                                            total_audio_copied += 1
                                        else:
                                            print(f"  Warning: Audio file not found: {source_audio}")
                                            total_audio_failed += 1
                                    except Exception as e:
                                        print(f"  Warning: Failed to copy audio {audio_path}: {e}")
                                        total_audio_failed += 1

                            else:
                                # Copy the generated audio folder if it exists
                                generated = entry.get('generated', {})
                                if generated.get('ok') and generated.get('turns'):
                                    audio_folder_path = str(Path(generated['turns'][0]['path']).parent)
                                    source_folder = os.path.join(input_dir, audio_folder_path)
                                    dest_folder = os.path.join(output_dir, audio_folder_path)
                                    # Copy the audio file
                                    try:
                                        shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
                                        total_audio_copied += 1
                                    except Exception as e:
                                        print(f"  Warning: Failed to copy audio folder {audio_folder_path}: {e}")
                                        total_audio_failed += 1
                    
                    except json.JSONDecodeError as e:
                        print(f"  Warning: Could not parse line: {e}")
                        continue
            
            # Write filtered entries to output file
            output_file = Path(manifests_output_dir) / jsonl_file.name
            
            if filtered_entries:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for entry in filtered_entries:
                        f.write(json.dumps(entry) + '\n')
                
                print(f"  Total entries: {entry_count}")
                print(f"  Filtered entries (speaker={speaker_id}): {len(filtered_entries)}")
                print(f"  Saved to: {output_file}")
                
                total_entries += entry_count
                total_filtered += len(filtered_entries)
            else:
                print(f"  No entries found with speaker_id='{speaker_id}'")
                total_entries += entry_count
        
        except Exception as e:
            print(f"  Error processing {jsonl_file.name}: {e}")
            continue

    # Copy reference audio files
    if speaker_id == 'all':
        # Copy all reference audio from VCTK
        vctk_references_input = os.path.join(input_dir, 'references', 'VCTK')
        vctk_references_output = os.path.join(output_dir, 'references', 'VCTK')
        if os.path.exists(vctk_references_input):
            shutil.copytree(vctk_references_input, vctk_references_output, dirs_exist_ok=True)
            print(f"\nCopied all VCTK reference audio files")
    else:
        # Copy only the specified speaker's reference audio
        shutil.copytree(os.path.join(input_dir, 'references', 'VCTK', f'{speaker_id}'), references_output_dir, dirs_exist_ok=True)
        print(f"\nCopied reference audio for speaker {speaker_id}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total entries processed: {total_entries}")
    if speaker_id == 'all':
        print(f"Total entries filtered (all speakers): {total_filtered}")
    else:
        print(f"Total entries filtered (speaker={speaker_id}): {total_filtered}")
    if max_files and max_files > 0:
        print(f"Maximum files processed: {max_files}")
    if max_entries and max_entries > 0:
        print(f"Maximum entries per file: {max_entries}")
    print(f"Total audio files copied: {total_audio_copied}")
    print(f"Total audio files failed: {total_audio_failed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter JSONL files by speaker ID'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/datasets/audios/COSAFE_MT/',
        help='Path to directory containing JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Safety_TTS_subset_p227_MT',
        help='Path to directory where filtered files will be saved'
    )
    parser.add_argument(
        '--speaker_id',
        type=str,
        default='p227',
        help='Speaker ID to filter by (default: p227). Use "all" to include all speakers.'
    )
    parser.add_argument(
        '--max_entries',
        type=int,
        default=None,
        help='Maximum number of entries per file (0 or omit for all)'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (0 or omit for all)'
    )

    parser.add_argument(
        '--is_multi_turn',
        type=bool,
        default=False,
        help='Indicate if the data is multi-turn'
    )
    
    args = parser.parse_args()
    
    filter_jsonl_by_speaker(args.input_dir, args.output_dir, args.speaker_id, args.max_entries, args.max_files, args.is_multi_turn)


if __name__ == "__main__":
    main() 
