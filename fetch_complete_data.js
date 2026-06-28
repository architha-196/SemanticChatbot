const fs = require("fs");
const https = require("https");
const { parse } = require("csv-parse/sync");

const SHEET_ID = "1DXrsx4g7WH-E-40YKPn1gLlh0exozlHbkVUonbL7ABg";
const SHEET_GID = "999155847";

const URL_ATTEMPTS = [
  `https://docs.google.com/spreadsheets/d/${SHEET_ID}/gviz/tq?tqx=out:csv&gid=${SHEET_GID}`,
  `https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv&gid=${SHEET_GID}`,
  `https://docs.google.com/spreadsheets/d/${SHEET_ID}/pub?output=csv&gid=${SHEET_GID}`,
  `https://docs.google.com/spreadsheets/d/${SHEET_ID}/gviz/tq?tqx=out:csv`,
  `https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv`,
];

console.log("Fetching complete student response data...");
console.log(`Sheet ID: ${SHEET_ID}`);
console.log(`Sheet GID: ${SHEET_GID}`);

async function fetchCompleteData() {
  for (let i = 0; i < URL_ATTEMPTS.length; i += 1) {
    const url = URL_ATTEMPTS[i];
    console.log(`\nAttempt ${i + 1}: ${url}`);

    try {
      const csvText = await fetchText(url);
      const records = parseCsv(csvText);

      if (records.length === 0) {
        throw new Error("No student records found");
      }

      const fieldCount = Object.keys(records[0]).length;
      if (fieldCount < 10) {
        throw new Error(`Only ${fieldCount} columns found; expected full assessment data`);
      }

      fs.writeFileSync("output.json", JSON.stringify(records, null, 2), "utf8");

      console.log(`Saved output.json with ${records.length} student records`);
      console.log(`Fields per student: ${fieldCount}`);
      console.log(`First student email: ${records[0]["Email Address"] || "Not found"}`);
      return { success: true, data: records };
    } catch (err) {
      console.log(`Failed attempt ${i + 1}: ${err.message}`);
    }
  }

  console.log("\nAutomatic fetch failed.");
  console.log("Manual fallback:");
  console.log("1. Open the Google Sheet");
  console.log("2. File -> Download -> Comma separated values (.csv)");
  console.log("3. Save it as student_responses.csv in this folder");
  console.log("4. Run: venv\\Scripts\\python.exe csv_to_json_converter.py");
  return { success: false };
}

function fetchText(url, redirectsLeft = 5) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, (response) => {
      const { statusCode, headers } = response;
      console.log(`Status: ${statusCode}`);

      if ([301, 302, 303, 307, 308].includes(statusCode)) {
        if (!headers.location || redirectsLeft === 0) {
          reject(new Error("Redirect failed"));
          return;
        }

        const nextUrl = new URL(headers.location, url).toString();
        response.resume();
        fetchText(nextUrl, redirectsLeft - 1).then(resolve, reject);
        return;
      }

      if (statusCode !== 200) {
        response.resume();
        reject(new Error(`HTTP ${statusCode}`));
        return;
      }

      let data = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        data += chunk;
      });
      response.on("end", () => resolve(data));
    });

    request.on("error", reject);
    request.setTimeout(15000, () => {
      request.destroy();
      reject(new Error("Request timeout"));
    });
  });
}

function parseCsv(csvText) {
  const rows = parse(csvText, {
    bom: true,
    columns: true,
    relax_column_count: true,
    skip_empty_lines: true,
    trim: true,
  });

  return rows.map((row) => {
    const cleaned = {};
    Object.entries(row).forEach(([key, value]) => {
      cleaned[key.trim()] = typeof value === "string" ? value.trim() : value;
    });
    return cleaned;
  });
}

fetchCompleteData()
  .then((result) => {
    if (result.success) {
      console.log("\nFetch complete. Run the backend with: venv\\Scripts\\python.exe chatbot1.py");
    }
  })
  .catch((err) => {
    console.log(`\nError: ${err.message}`);
  });
