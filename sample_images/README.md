# sample_images/

Place your test images here to use them with the CLI or for manual testing.

## Suggested test images (one per category)

| Filename            | Category      | Source suggestion                          |
|---------------------|---------------|--------------------------------------------|
| person.jpg          | Person        | Any royalty-free portrait                  |
| animal.jpg          | Animal        | Dog, cat, bird, etc.                       |
| food.jpg            | Food          | A meal or dish photo                       |
| street.jpg          | Street scene  | Urban / outdoor scene                      |
| object.jpg          | Object        | A single household item                    |

## Notes

- Accepted formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- Maximum file size: **10 MB**
- The app will auto-download `sample_cat.jpg` on first run if you are online.
  That file is listed in `.gitignore` so it won't be committed to your repo.

## CLI usage

```bash
python cli.py sample_images/animal.jpg
python cli.py sample_images/street.jpg --beams 6 --max-length 80
```
