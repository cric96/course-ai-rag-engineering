-- Match source headings in document order, including duplicate slide titles.
-- Read the current files at render time so GitHub line anchors cannot go stale.
local repository = "https://github.com/cric96/course-ai-rag-engineering/blob/main/"

local function lines(path)
  local result = {}
  for line in io.lines(path) do result[#result + 1] = line end
  return result
end

local function source_slides()
  local result = {}
  local main = io.open("corso.qmd")
  local manifest = main and "corso.qmd" or "_source-links.qmd"
  if main then main:close() end
  for _, line in ipairs(lines(manifest)) do
    local path = line:match("{{<%s*include%s+([^%s>]+)%s*>}}")
    if path then
      local source = lines(path)
      local fence, fence_length
      local previous
      local divs = {}
      for number, text in ipairs(source) do
        local marker = text:match("^%s*(```+)") or text:match("^%s*(~~~+)")
        if marker then
          if not fence then
            fence, fence_length = marker:sub(1, 1), #marker
          elseif marker:sub(1, 1) == fence and #marker >= fence_length
              and text:match("^%s*[`~]+%s*$") then
            fence = nil
          end
        elseif not fence then
          local colons, rest = text:match("^%s*(:+)(.*)$")
          if colons and #colons >= 3 then
            if rest:match("%S") then
              divs[#divs + 1] = #colons
            else
              while #divs > 0 and divs[#divs] <= #colons do
                local length = table.remove(divs)
                if length == #colons then break end
              end
            end
          end
          local hashes = text:match("^(#+)%s+")
          if hashes and #hashes <= 2 and #divs == 0 then
            if previous then previous.last = number - 1 end
            local heading = pandoc.read(text, "markdown").blocks[1]
            previous = {
              path = "slides/" .. path, first = number, last = #source,
              title = pandoc.utils.stringify(heading.content), level = #hashes,
            }
            result[#result + 1] = previous
          end
        end
      end
    end
  end
  return result
end

local function link(label, path)
  return pandoc.Link(label, repository .. path, "", pandoc.Attr("", {}, {
    target = "_blank", rel = "noopener noreferrer",
  }))
end

function Pandoc(doc)
  if not FORMAT:match("revealjs") then return doc end
  local sources, position = source_slides(), 0
  local blocks = pandoc.List()
  for _, block in ipairs(doc.blocks) do
    local footer
    if block.t == "Header" and block.level <= 2 then
      position = position + 1
      local source = sources[position]
      assert(source and source.level == block.level
          and source.title == pandoc.utils.stringify(block.content),
        "Source links: slide/source heading mismatch at slide " .. position)
      local items = pandoc.List({link("Sorgente slide", string.format(
        "%s#L%d-L%d", source.path, source.first, source.last
      ))})
      local count = 0
      for path in (block.attributes["data-code"] or ""):gmatch("[^,%s]+") do
        count = count + 1
        assert(count <= 3, "Source links: at most three implementation links per slide")
        items:insert(pandoc.Str(" · "))
        items:insert(link(path, path))
      end
      -- Keep implementation metadata out of the resulting slide attributes.
      block.attributes["data-code"] = nil
      footer = pandoc.Div({pandoc.Para(items)}, pandoc.Attr("", {
        "slide-code-links",
      }, {["data-title-source"] = repository .. "slides/corso.qmd"}))
    end
    blocks:insert(block)
    if footer then blocks:insert(footer) end
  end
  assert(position == #sources, "Source links: unmatched source headings")
  doc.blocks = blocks
  return doc
end
