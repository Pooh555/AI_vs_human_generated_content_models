#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 \"source_directory\" \"destination_directory\" percentage"
  echo "Example: $0 \"/path/to/your/directory1\" \"/path/to/your/directory2\" 5"
  exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"
PERCENTAGE="$3"

if ! [[ "$PERCENTAGE" =~ ^[0-9]+$ ]]; then
  echo "Error: Percentage must be a valid integer."
  exit 1
fi

if [ "$PERCENTAGE" -lt 0 ] || [ "$PERCENTAGE" -gt 100 ]; then
  echo "Error: Percentage must be between 0 and 100."
  exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory '$SOURCE_DIR' not found."
  exit 1
fi

if [ ! -d "$DEST_DIR" ]; then
  echo "Destination directory '$DEST_DIR' not found. Creating it."
  mkdir -p "$DEST_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Could not create destination directory '$DEST_DIR'."
    exit 1
  fi
fi

mapfile -d '' all_files < <(find "$SOURCE_DIR" -maxdepth 1 -type f -print0)

total_files=${#all_files[@]}

if [ "$total_files" -eq 0 ]; then
  echo "No files found in the source directory '$SOURCE_DIR'."
  exit 0
fi

files_to_move=$(printf "%.0f" $(echo "$total_files * $PERCENTAGE / 100" | bc -l))

if [ "$files_to_move" -eq 0 ] && [ "$total_files" -gt 0 ] && [ "$PERCENTAGE" -gt 0 ]; then
  files_to_move=1
fi

if [ "$files_to_move" -gt "$total_files" ]; then
  files_to_move="$total_files"
fi

echo "Total files in '$SOURCE_DIR': $total_files"
echo "Attempting to move $files_to_move files ($PERCENTAGE%)."
readarray -t selected_files < <(printf "%s\n" "${all_files[@]}" | shuf | head -n "$files_to_move")

if [ "${#selected_files[@]}" -eq 0 ]; then
  echo "No files selected for moving."
else
  echo "Moving selected files to '$DEST_DIR':"
  for file_path in "${selected_files[@]}"; do
    filename=$(basename "$file_path")
    echo "Moving: '$filename'"
    mv "$file_path" "$DEST_DIR/$filename"
    if [ $? -ne 0 ]; then
      echo "Error: Could not move file '$file_path'."
    fi
  done
fi

echo "Script finished."