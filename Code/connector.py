import pathway as pw


# Assuming this is the code that loads the Google Drive file
try:
   table = pw.io.gdrive.read(
       object_id="17PGcwSLjQJ-wg7L_WbvfRqkNQSNe-_sY",
       service_user_credentials_file="credentials.json"
   )
   print("Data loaded successfully")
except Exception as e:
   print(f"Error loading data: {e}")


pw.run()
print("Pathway run completed")

