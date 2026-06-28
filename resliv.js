const fs = require("fs");

async function fetchGoogleSheet(sheetID, sheetGid) {
    const url = `https://docs.google.com/spreadsheets/d/${sheetID}/gviz/tq?tqx=out:json&gid=${sheetGid}&cache_bust=${Date.now()}`;

  try {
    const response = await fetch(url); // No need for require('node-fetch')
    const text = await response.text();
    const jsonText = text.substring(47, text.length - 2); // Clean response
    const data = JSON.parse(jsonText);

    let headers = data.table.cols.map((col, index) => col.label || `Column ${index + 1}`);
    let jsonData = data.table.rows.map(row => {
      let rowData = {};
      headers.forEach((header, index) => {
        const cell = row.c[index];
        rowData[header] = cell ? (cell.f || cell.v || "") : "";
      });
      return rowData;
    }).filter(row => Object.values(row).some(value => String(value).trim() !== ""));

    fs.writeFileSync("output.json", JSON.stringify(jsonData, null, 2));
    console.log("✅ JSON file created: output.json");

  } catch (error) {
    console.error("❌ Error fetching Google Sheet:", error);
  }
}

// Replace with your Google Sheet ID and Sheet Name
const SHEET_ID = "1DXrsx4g7WH-E-40YKPn1gLlh0exozlHbkVUonbL7ABg"; // Replace with your actual Sheet ID
const SHEET_GID = "1324852220";

fetchGoogleSheet(SHEET_ID, SHEET_GID);
