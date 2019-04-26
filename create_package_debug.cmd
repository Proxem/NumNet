echo OFF
echo[
echo[
echo                   ##########################
echo                   ##   Building NumNet    ##
echo                   ##########################
echo[
echo[

dotnet build -c Debug

echo NumNet was built in debug configuration, ready to create package
pause
echo[
echo[
echo                 #################################
echo                 ##   Building NumNet Package   ##
echo                 #################################
echo[
echo[

nuget pack Proxem.NumNet/nuspec/Proxem.NumNet.Debug.nuspec -Symbols -OutputDirectory Proxem.NumNet/nuspec/

echo Package was created in Debug mode in folder Proxem.NumNet/nuspec/
pause