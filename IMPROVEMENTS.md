# TTE Downloader Improvements

## Key Issues Fixed

### 1. **URL Construction (CRITICAL)**
**Problem:** Your original script manually constructed download URLs:
```python
download_url = f"https://zenodo.org/records/{record_id}/files/{filename}"
```

**Solution:** The improved version uses the URLs provided by Zenodo's API:
```python
download_url = file_info.get("links", {}).get("self")
```

This is the **primary cause** of your "forbidden" errors. Zenodo expects you to use the URLs it provides in the API response, not manually constructed ones.

### 2. **Rate Limiting**
**Problem:** No delays between requests, which can trigger rate limits.

**Solution:** Added a rate limiting mechanism:
- Minimum 1 second between API requests
- Prevents overwhelming Zenodo's servers
- Reduces chance of getting blocked

### 3. **Better Error Handling**
**Problem:** Errors could cause the script to crash or leave partial downloads.

**Solution:**
- Try-catch blocks around all network operations
- Temporary files during download (`.download` suffix)
- Only rename to final name after successful download
- Keeps bad files for inspection instead of crashing

### 4. **Improved File Checking**
**Problem:** Script checked if zip files exist, but not if they've been extracted.

**Solution:**
- For zip files: checks if the extracted directory exists
- Skips re-downloading and re-extracting if data is already present
- More efficient resume capability

### 5. **Better Progress Reporting**
**Problem:** Limited feedback during long downloads.

**Solution:**
- Shows progress every 10MB during download
- File counter (e.g., "[3/25] Processing tte_2015.zip")
- Clear status messages (✓ for success, ✗ for errors)

## Usage

### Basic Usage
```python
from improved_tte_downloader import download_tte_dataset

download_tte_dataset(
    from_date="2010-01-01",
    to_date="2015-12-31",
    remote_url="https://doi.org/10.5281/zenodo.18235818",
    local_output_dir="./tte_dataset",
    force_update=False
)
```

### Advanced Usage
```python
from improved_tte_downloader import TTEDownloader

downloader = TTEDownloader("./my_data")

# Download with force update (re-download everything)
downloader.download_tte_data(
    from_date="2020-01-01",
    to_date="2023-12-31",
    remote_url="https://doi.org/10.5281/zenodo.18235818",
    force_update=True
)
```

## How It Works

1. **Fetch Manifest:** Queries Zenodo API for file list
2. **Filter by Year:** Only includes years in your date range
3. **Check Local Files:** Determines what's already downloaded
4. **Download Missing Files:** Downloads only what's needed
5. **Extract Archives:** Unzips files and removes zip archives
6. **Load Data:** Reads CSV files from extracted directories
7. **Filter by Date:** Applies exact date range filter to patents
8. **Deduplicate:** Removes duplicate patent entries
9. **Save Results:** Saves final filtered dataset

## File Structure

After running, your directory will look like:
```
tte_dataset/
├── local_manifest.json          # Tracks downloaded files
├── tte/
│   ├── dataset_manifest.json    # Dataset metadata
│   ├── tte_2010/                # Extracted year directories
│   │   ├── patent_text_2010.csv
│   │   ├── matches_2010.csv
│   │   └── patent_embed_2010.npy
│   ├── tte_2011/
│   │   └── ...
│   └── tte_models/              # ML models (if downloaded)
└── patents/
    └── patent_text.csv          # Final merged dataset
└── matches.csv                   # Final merged matches
```

## Comparison with zget

### When to Use This Script
- ✓ You only need specific years
- ✓ You want date-range filtering
- ✓ You want merged output files
- ✓ You need custom processing logic

### When to Use zget
- ✓ You want the entire dataset
- ✓ You don't need date filtering
- ✓ You prefer command-line tools
- ✓ You're just exploring the data

### Using zget
If you want to try zget instead:
```bash
# Install zget
pip install zget

# Download entire record
zget 10.5281/zenodo.18235818

# Download specific files
zget 10.5281/zenodo.18235818 -f tte_2020.zip tte_2021.zip
```

Then you could use a lighter script to just merge and filter the data.

## Troubleshooting

### Still Getting "Forbidden" Errors?
1. Check your internet connection
2. Verify the DOI is correct
3. Try increasing the `min_request_interval` in the code (currently 1.0 seconds)
4. Check if Zenodo is having issues: https://status.zenodo.org/

### Downloads Are Slow?
This is normal for large files (200-450MB per year). The script shows progress every 10MB.

### Script Crashes During Download?
The improved version saves progress. Just run it again - it will skip already-downloaded files.

### Need to Start Over?
Delete the `tte/` subdirectory and `local_manifest.json`, then run again.

## Performance Tips

1. **Download overnight:** For large date ranges (10+ years), this can take hours
2. **Use fast internet:** Large files benefit from high-speed connections
3. **Don't force_update unless needed:** This re-downloads everything
4. **Keep the tte/ directory:** Speeds up subsequent runs with different date ranges

## Migration from Old Script

Your old script should work with these steps:

1. **Replace the file:** Use `improved_tte_downloader.py` instead
2. **Update imports:** Change import statements if needed
3. **Same function signature:** `download_tte_dataset()` works the same way
4. **Check output:** Results are saved in the same locations

No changes needed to downstream code that reads the CSV files!
