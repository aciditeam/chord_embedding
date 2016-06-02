require 'torch'

data=torch.load("../final_data/event_total.csv",'ascii')
path="../final_data/event_total2.csv"

function write_tensor(path, data, sep)
    local out = assert(io.open(path, "w")) -- open a file for serialization
    splitter = ","
    for i=1,data:size(1) do
        for j=1,data:size(2) do
            out:write(data[i][j])
            if j == data:size(2) then
                out:write("\n")
            else
                out:write(splitter)
            end
        end
    end
    out:close()
end

function write_table(path, data, sep)
    sep = sep or ','
    local file = assert(io.open(path, "w"))
    for i=1,#data do
        for j=1,#data[i] do
            if j>1 then file:write(sep) end
            file:write(data[i][j])
        end
        file:write('\n')
    end
    file:close()
end

function write_table_1D(path, data, sep)
    sep = sep or ','
    local file = assert(io.open(path, "w"))
    for i=1,#data do
        if i>1 then file:write(sep) end
        file:write(data[i])
    end
    file:close()
end


-- write_tensor(path, data, ",")
-- write_table(path, data, ",")
write_table_1D(path, data, ",")
