local MIDI = require 'MIDI'
local lfs=require 'lfs'


math.randomseed(os.time())

local function deepcopy(object)  -- http://lua-users.org/wiki/CopyTable
	local lookup_table = {}
	local function _copy(object)
		if type(object) ~= "table" then
			return object
		elseif lookup_table[object] then
			return lookup_table[object]
		end
		local new_table = {}
		lookup_table[object] = new_table
		for index, value in pairs(object) do
			new_table[_copy(index)] = _copy(value)
		end
		return setmetatable(new_table, getmetatable(object))
	end
	return _copy(object)
end


--[[ fh = assert(io.open('Midi2/10.mid', "rb"))
  local midi = fh:read('*all')
  fh:close()
local score = MIDI.midi2score(midi) ]]--


local function round(x)
return math.floor(x+0.5)
end



local function import_data(score, division)
  -- Import the score statistics
  local stats = MIDI.score2stats(score)
  -- Data matrix
  local array = {};
  local name = {};
	-- Pitch ranges by track
	local pitch_ranges = {};
  -- Name of the tracks
  local track_labels={};
  -- This represents how much ticks will be one step
  local quantifier = stats["ticks_per_quarter"] / (division / 4);
  -- The actual length of the track in numbers of steps
  local length = round(stats["nticks"] / quantifier) + 1;
	-- Event vector
	local event_vector=torch.Tensor(length):zero()
  -- Parse the whole score
	local h=1
  for k = 2,#score do
    if (#score[k] <= 9) then goto skip_track ; end
	  if (score[k][2][1]=='track_name') then
	    name = tostring(score[k][2][3]);

			-- Suppressing instrument name redundancy
			if name == "Kboard 2 (Pn2 1)" or name == "Kboard 2 (Pn2)" then
				name = "Kboard 2"
			elseif name == "Kboard 3 (Celesta)" then
				name= "Kboard 3"
			elseif name=="Kboard 1 (Pno 1)" then
				name ="Kboard 1"
			elseif name== "Bass Trb (Tuba MB)" or name == "Bass Trombone" then
				name = "Trombone Bass"
			elseif name == "Bassoon Contra 1" then
				name = "Contrabassoon"
			elseif name == "Bass Clarinet" or name == "Bass Clarinet (Cl3 MB)" then
				name = "Clarinet Bass 1"
			elseif name == "Double bass b" or name == "Double bass Tutti"   then
				name = "Double Bass Tutti"
			elseif name == "Double bass Solo" then
				name= "Double bass Solo 1"
			elseif name == "Violin I Tutti" or name == "Violins Ib" then
				name="Violins I Tutti"
			elseif name == "Violin II Tutti" then
				name ="Violins II Tutti"
			elseif name == "Violin I solo 1" then
				name ="Violin I solo"
			elseif name == "Violin II solo 1" then
				name = "Violin II solo"
			elseif name =="Cello Tutti" or name == "Celli b" then
				name = "Celli Tutti"
			elseif name == "Tuba" or name == "Tuba 1 (Harp MB)" then
				name = "Tuba 1"
			elseif name == "English Horn" then
				name =  "Oboe English Horn 1"
			elseif name == "Flute Picc 1" or name == "Piccolo" then
				name = "Piccolo 1"
			end
			-- Instantiate each track _once_ and in the FFI-C memory space
      array[name] = torch.Tensor(127, length);
      -- Set whole array to zero
      array[name]:zero();
			-- Create a pitch range by track
			pitch_ranges[name] = torch.Tensor(2);
			pitch_ranges[name][1] = 512;
			pitch_ranges[name][2] = -1;
      -- Record the name of the track
      track_labels[h] = name;
			h=h+1
    end
    -- Parse the whole score
    for i = 1,#score[k] do
      -- Current event
	    local event = score[k][i];
	    -- Just parse
	    if (event[1]=='note') then
	      -- Onset
	      local onset = (round(event[2] / quantifier) + 1)
	      local event_length = round(event[3] / quantifier)
	      local pitch = event[5]
	      local vel = event[6]
				pitch_ranges[name][1] = math.min(pitch,pitch_ranges[name][1]);
				pitch_ranges[name][2] = math.max(pitch,pitch_ranges[name][2]);;
	      -- Fill corresponding steps
	      for j = onset,onset+event_length-1 do array[name][pitch][j] = vel; end
				event_vector[onset]=1
				event_vector[onset+event_length]=1
	    end
	  end
	  ::skip_track::
  end
	-- Return the complete array, the track labels and the pitch_ranges
  return array, track_labels, pitch_ranges,event_vector
end


-- Suppress velocity in the data, 0 or 1 : Note or no note
local function binarize(array,track_labels)
  local num_track=#track_labels
  local length=#array[track_labels[1]][1]

  for i=1,num_track
  do
    for j=0,127
    do
      for k=0,length
      do
        if (array[track_labels[i]][j][k]>0)
        then
          array[track_labels[i]][j][k]=1
        end
      end
    end
  end
  return array
end

--array,track_labels=import_data(score,24)


--
local function array2csv(array,track_labels,nom_fichier,score)

  local quantization=4
  local num_track=#track_labels
  local length=#array[track_labels[1]][1]
  local to_treat=deepcopy(array)
	note_vector={}
	local l=1

	for i=1,2  -- #num_track
  do

		--[[name=tostring(track_labels[i])
		note_vector[name]={}
		for j=0,127 do
			 note_vector[name][j] = {}
		end]]--

		name=tostring(track_labels[i])

    local csvfile=assert(io.open(nom_fichier .. tostring(track_labels[i]) .. '.csv','w'))
    io.output(csvfile)
    io.write('')
    io.close(csvfile)

    local csvfile=assert(io.open(nom_fichier .. tostring(track_labels[i]) .. '.csv','a'))
    io.output(csvfile)
    io.write('t0',',','dt' , ',' ,'pitch' ,',','dyn',',','quantization','\n')
    for j=0,127
    do
      for k=0,length
      do
        if (to_treat[track_labels[i]][j][k]~=0)
        then
          local dt=1
          local dyn=to_treat[track_labels[i]][j][k] -- ou pas normalise
          io.write(k,',')
          while(to_treat[track_labels[i]][j][k+dt]~=0)
          do
            to_treat[track_labels[i]][j][k+dt]=0
            dt=dt+1
          end
          io.write(dt,',',j,',',dyn,',',quantization,'\n')
					note_vector[l]={name,k,dt,j}
					--track,begin,length,pitch
					l=l+1
        end
      end
    end
    io.close(csvfile)
  end
	return note_vector
end

--note_vector=array2csv(array,track_labels,'CSV/CSV/nom_fichier',score)
--print(note_vector[1])



--[[for a,b in pairs(track_labels) do
  print('num =', a, 'name =', b)
end

print('\n')
print(#array) ]]--

--[[
for track, channel in pairs(stats['patch_changes_by_track']) do
  for channel, instru in pairs(stats['patch_changes_by_track'][track]) do
    print('track =', track, ' channel =', channel, 'instru=', instru)
end
end

print('\n')

print(stats['patch_changes_total'][4])

print('\n')

for track, pitch in pairs(stats['pitch_range_by_track']) do
  print('track=',track,'pitches=', pitch[1], pitch[2])
end

print('nticks= ', stats['nticks'])

]]--


--array[name][pitch][time]


local function transpose(array, track_labels, howmuch, score)
--Transpose vers le haut pour howmuch>0
  local num_track=#track_labels
  local length=#array[track_labels[1]][1]
  local stats= MIDI.score2stats(score)
  transposed={}

  for k=1,num_track do

        name=tostring(track_labels[k])
        transposed[name]={}
        for i=0,127 do
           transposed[name][i] = {}
        end

      if (howmuch>0)
      then
        for i=howmuch,127
        do
          for j=0,length
          do
            transposed[track_labels[k]][i][j]=array[track_labels[k]][i-howmuch][j]

          end
        end
      else
        for i=0,127+howmuch
        do
          for j=0,length
          do
            transposed[track_labels[k]][i][j]=array[track_labels[k]][i-howmuch][j]
          end
        end
      end
  end
  return transposed
end

--transposed=transpose(array,track_labels,1,score)


--note_vector_transposed=array2csv(transposed,track_labels,'CSV/CSV/nom_fichier_transposed',score)


local function suppress_instru(array,track_labels)
	local array_sup=deepcopy(array)

	array_sup[track_labels[math.random(1,#track_labels)]]=nil
	return array_sup
end

--array_sup=suppress_instru(array,track_labels)


local function suppress_note(array,track_labels,note_vector)
	local array_note_sup=deepcopy(array)
	--note_vector[l]={name,k,dt,j}
	--track,begin,length,pitch
	local a_sup=note_vector[math.random(1,#note_vector)]
	for i=a_sup[2],a_sup[2]+a_sup[3]
	do
		array_note_sup[a_sup[1]][a_sup[4]][i]=0
	end
return a_sup,array_note_sup
end

--a_sup,array_note_sup=suppress_note(array,track_labels,note_vector)

local function alter_note(array,track_labels,note_vector)
	local array_note_alt=deepcopy(array)
	--note_vector[l]={name,k,dt,j}
	--track,begin,length,pitch
	local a_alt=note_vector[math.random(1,#note_vector)]
	local b={-1,1}
	local c=b[math.random(1,2)]
	for i=a_alt[2],a_alt[2]+a_alt[3]
	do
		array_note_alt[a_alt[1]][a_alt[4]][i]=0
		array_note_alt[a_alt[1]][a_alt[4]+c][i]=1
	end
	return a_alt,array_note_alt
end

--a_alt,array_note_alt=suppress_note(array,track_labels,note_vector)


local function keep_harmony(array,track_labels,howmuch,score,nbnote)
  local num_track=#track_labels
  local length=#array[track_labels[1]][1]
  local harmony_array={}
  local harmony_vector={}

  --definir harmony_array

  local harmony_length=0

  for i=0,length
  do
    sum=0
    for j=1,num_track
    do
      for k=0, 127
      do
        if (array[track_labels[j]][k][i]==1)
        then
          harmony_vector[j][k]=1
          sum=sum+1
        end
      end
    end
    if (sum>nbnote)
    then

      --remplir la harmony_array, penser au cas du piano


      harmony_length=harmony_length+1

    end
  end
end


track_list={}
i=1
DIR_SEP="/" --should be "/" for Unix platforms (Linux and Mac)

function browseFolder_read(root)
	for entity in lfs.dir(root) do
		collectgarbage();
		if entity~="." and entity~=".." and entity~=".DS_Store" then
			local fullPath=root..DIR_SEP..entity
			--print("root: "..root..", entity: "..entity..", mode: "..(lfs.attributes(fullPath,"mode")or "-")..", full path: "..fullPath)
			local mode=lfs.attributes(fullPath,"mode")
			if mode=="file" then
				--this is where the processing happens. I print the name of the file and its path but it can be any code
				print(root.." > "..entity)
				track_list[i]=fullPath  --record the path

				i=i+1
			elseif mode=="directory" then
				browseFolder_read(fullPath);
			end
		end
	end
end


total_track={}
count={}
tracing={}
root='Midi'
browseFolder_read(root)
--for j=1,#total_track do
--	print(total_track[j],count[total_track[j]])
--end

--[[
  stats["num_notes_by_channel"];    -- Counts of number of notes per channel
  stats["pitches"];                 -- Counts of number of notes per MIDI pitch value (not on channel 9 ?!)
  stats["nticks"];                  -- Complete number of ticks
  stats["ntracks"];                 -- Number of tracks
  stats["ticks_per_quarter"];       -- Number of ticks per quarter note
  stats["percussion"]               -- A dictionary histogram of channel-9 events
  stats["pitch_range_sum"];         -- Sum over tracks of the pitch_ranges
  stats["bank_select"];             -- Array of 2-element arrays {msb,lsb}
  stats["channels_by_track"];       -- Table of which MIDI channels are used by each track
  stats["channels_total"];          -- Number of track that contain a given MIDI channel
  stats["pitch_range_by_track"];    -- For each track, gives the minimum and maximum pitches
  stats["general_midi_mode"];       -- Array
  stats["patch_changes_by_track"];  -- Table of tables
  stats["patch_changes_total"];     -- Array
  ]]--


-- Defining the overall structures
array_total={}
track_labels_total={}
length_total=0
time_pointer={}
time_pointer[0]=0
pitch_range_by_track={} -- Pitch range per instrument
event_total={}
print('Memory pre-import' .. collectgarbage("count"));
for i=1,#track_list do
	--Openning/Reading  (24 tick per beat)
	fh = assert(io.open(track_list[i], "rb"))
  local midi = fh:read('*all')
	fh:close()
  print('Importing ' .. track_list[i]);
	local score = MIDI.midi2score(midi)
	local stats= MIDI.score2stats(score)
	array,track_labels,pitch_ranges,event_vector=import_data(score,24)
	-- If we want to make data augmenation
	-- It's here

	local num_track=#track_labels
	local length=array[track_labels[1]]:size(2);
	length_total=length_total+length
	-- Midi data storage
	array_total[i]=array
	-- Tracklabels per song storage
	track_labels_total[i]=track_labels
	-- Beginning/ End per song storage
	time_pointer[i]=length_total

	for h=time_pointer[i-1]+1,time_pointer[i] do
		event_total[h]=event_vector[h-time_pointer[i-1]]
	end
	--print(time_pointer[i-1]+1,time_pointer[i])
	--print(length)

	-- Defining all the instruments present in all songs
	-- Defining pitch range of each instrument in all songs
	local j=0
	for h,u in pairs(track_labels) do
		for g,v in pairs(total_track) do
			if v == u then
				j=j+1  -- Already exists or not?
				count[u]=count[u]+1

				if pitch_range_by_track[g]==nil then
					pitch_range_by_track[g]=pitch_ranges[u]
				end

				-- Adapting pitch range of one instrument
				if pitch_ranges[u][1]<pitch_range_by_track[g][1] then
					pitch_range_by_track[g][1]=pitch_ranges[u][1]
				end
				if pitch_ranges[u][2]>pitch_range_by_track[g][2] then
					pitch_range_by_track[g][2]=pitch_ranges[u][2]
				end
			end
		end
		if j==0 then  -- Instrument not registered yet
			total_track[#total_track+1]=u
			count[u]=1  -- Instrument appears first
			tracing[u]=track_list[i] -- Appears first in this song
			if pitch_range_by_track[g]==nil then
				pitch_range_by_track[#total_track]=pitch_ranges[u]
			end
		end
		j=0
	end
end
print('Memory post-import ' .. collectgarbage("count"));

print(track_list)
print(time_pointer)

--print(total_track)
--print(pitch_range_by_track)

pitch_indice=0  -- Total size of massive_matrix in pitch
pitch_per_inst={} -- Beginning/end of pitch for each instrument
for i,u in pairs(total_track) do
	pitch_indice=pitch_indice+1

	--If no note at all were played in an instrument
	if pitch_range_by_track[i][1] == 512 then
		pitch_range_by_track[i][1]=0
		pitch_range_by_track[i][2]=0
	end
	pitch_per_inst[i]={}
	pitch_per_inst[i][1]=pitch_indice  -- Beginning of range
	pitch_indice=pitch_indice+pitch_range_by_track[i][2]-pitch_range_by_track[i][1]
	pitch_per_inst[i][2]=pitch_indice  -- End of range
	--print(pitch_per_inst[i][1])
	--print(pitch_per_inst[i][2])

end
-- 128*#total_track = 14720 pour info
--print(length_total*pitch_indice) --Taille de la matrice

-- Checking proportion of events
--[[function sum(t)
    local sum = 0
    for k,v in pairs(t) do
        sum = sum + v
    end

    return sum
end
sum=sum(event_total)
prop=sum/#event_total
print(prop)]]--

--Matrix of all instrument all songs

--massive_matrix=torch.Tensor(pitch_indice,length_total):zero()
-- Filling the matrix with the data
for i=1,#track_list do
	for l=1,#track_labels_total[i] do
		for j=time_pointer[i-1]+1,time_pointer[i] do
			for k=pitch_per_inst[l][1],pitch_per_inst[l][2] do
				--massive_matrix[k][j]= array_total[i][track_labels_total[i][l]][k-pitch_per_inst[l][1]+1][j-time_pointer[i-1]]
			end
		end
	end
end

-- Saving the massive_matrix, time_pointer , pitch_per_inst, track_labels_total, event_total

 --torch.save("final_data/massive_matrix.csv",massive_matrix,'ascii')
 --torch.save("final_data/time_pointer.csv",time_pointer,'ascii')
 --torch.save('final_data/pitch_per_inst.csv',pitch_per_inst,'ascii')
 --torch.save('final_data/track_labels_total.csv',track_labels_total,'ascii')
 --torch.save('final_data/event_total.csv',event_total,'ascii')
