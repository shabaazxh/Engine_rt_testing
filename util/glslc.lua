-- Note: known bug: make clean won't delete the generated .spv files.
--
-- Note: limitation: this doesn't check for the architecture, so unsure what
-- a M1 mac will do.

local binname = nil;
local shaderc = "third_party/shaderc";

local host = os.host();
local override = os.getenv( "COMP5892M_HOST_OVERRIDE" );
if override then
	host = override;
	print( "COMP5892M_HOST_OVERRIDE: '" .. host .. "'" );
end

if "windows" == host then
	binname = "win-x86_64/glslc.exe";
elseif "linux" == host then
	binname = "linux-x86_64/glslc";
elseif "macosx" == host then
	binname = "macos-x86_64/glslc";
else
	error( "No glslc binary for this platform (" .. host .. ")" );
end

local glslc = path.join( shaderc, binname );

local glslc_build_command_ = function( kind, ext, opt, opath, ipaths )
	local istr = "";
	for _,ipath in ipairs(ipaths) do
		if "/" == ipath:sub(1,1) then
			istr = istr .. "\"-I" .. ipath .. "\"";
		else
			istr = istr .. "\"-I%{wks.location}/" .. ipath .. "\"";
		end
	end
	istr = istr .. " ";

	local ofile, odir = "", "";
	if "/" == opath:sub(1,1) then
		odir = opath;
		ofile = ofile .. opath;
	else
		odir = "%{wks.location}/" .. opath;
		ofile = ofile .. "%{wks.location}/" .. opath;
	end
	ofile = ofile .. "/%{file.name}.spv";

	filter( "files:**." .. ext )
		buildmessage( "GLSLC: [" .. kind .. "] '%{file.name}'" );
		buildcommands( "{mkdir} \"" .. odir .. "\"" );
		buildcommands(
			 "\"%{wks.location}/" .. glslc ..  "\" "
			 .. opt .. " -g -O0" 
			 .. istr 
			 .. "-o \"" .. ofile .. "\" "
			 .. "\"%{file.relpath}\""
		)
		buildoutputs( ofile )
	filter "*"
end

handle_glsl_files = function( opt, opath, ipaths )
	local types = {
		{ "VERT", "vert" },
		{ "FRAG", "frag" },
		{ "COMP", "comp" },
		{ "GEOM", "geom" },
		{ "TESC", "tesc" },
		{ "TESE", "tese" },
		{ "RGEN", "rgen" },   -- Ray generation shader
		{ "RINT", "rint" },   -- Intersection shader
		{ "RAHIT", "rahit" }, -- Any-hit shader
		{ "RCHIT", "rchit" }, -- Closest-hit shader
		{ "RMISS", "rmiss" }, -- Miss shader
		{ "RCALL", "rcall" }  -- Callable shader
	};

	for _,ty in ipairs(types) do
		glslc_build_command_( ty[1], ty[2], opt, opath, ipaths )
	end
end

--EOF vim:syntax=lua:foldmethod=marker:ts=4:noexpandtab: 
