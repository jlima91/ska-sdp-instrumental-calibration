function setAllSectionsOpen(open) {
  document.querySelectorAll("details").forEach(function (section) {
    section.open = open;
  });
}

// Compare two table cells by their data-sort-value: as numbers when both
// are numbers, else as text. Empty values (not available) always go last.
function compareCells(cellA, cellB, ascending) {
  var a = cellA.dataset.sortValue;
  var b = cellB.dataset.sortValue;
  if (a === "" || b === "") {
    return (a === "") - (b === "");
  }
  var order = isNaN(a) || isNaN(b)
    ? a.localeCompare(b)
    : Number(a) - Number(b);
  return ascending ? order : -order;
}

// Make a .table-card (see data_table in components.html) paginated and
// sortable by clicking its column headers.
function initTable(card) {
  var body = card.querySelector("tbody");
  var headers = card.querySelectorAll("th");
  var pageSize = Number(card.dataset.pageSize);
  var itemName = card.dataset.itemName;
  var page = 0;

  function showPage(newPage) {
    var rows = body.rows;
    var npages = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.min(Math.max(newPage, 0), npages - 1);

    var first = page * pageSize;
    var last = Math.min(first + pageSize, rows.length);
    for (var i = 0; i < rows.length; i++) {
      rows[i].hidden = i < first || i >= last;
    }
    card.querySelector(".page-info").textContent =
      "Showing " + (rows.length ? first + 1 : 0) + " to " + last +
      " of " + rows.length + " total " + itemName;
  }

  function sortBy(columnIndex) {
    var header = headers[columnIndex];
    var ascending = !header.classList.contains("sorted-ascending");
    headers.forEach(function (other) {
      other.classList.remove("sorted-ascending", "sorted-descending");
    });
    header.classList.add(ascending ? "sorted-ascending" : "sorted-descending");

    Array.from(body.rows)
      .sort(function (rowA, rowB) {
        return compareCells(
          rowA.cells[columnIndex], rowB.cells[columnIndex], ascending
        );
      })
      .forEach(function (row) {
        body.appendChild(row);
      });
    showPage(0);
  }

  headers.forEach(function (header, columnIndex) {
    header.addEventListener("click", function () {
      sortBy(columnIndex);
    });
  });
  card.querySelector(".previous-page").addEventListener("click", function () {
    showPage(page - 1);
  });
  card.querySelector(".next-page").addEventListener("click", function () {
    showPage(page + 1);
  });
  showPage(0);
}

document.querySelectorAll(".table-card").forEach(initTable);
