-- Prefix every slide id with its chapter number: #/03-analisi-... in the address.
-- Chapters are the level-1 headings; the opening slides before the first one are 00.
function Pandoc(doc)
  local chapter = 0
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level <= 2 then
      if block.level == 1 then
        chapter = chapter + 1
      end
      block.identifier = string.format("%02d-%s", chapter, block.identifier)
    end
  end
  return doc
end
