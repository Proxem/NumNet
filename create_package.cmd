echo OFF
echo[
echo[
echo                   ##########################
echo                   ##   Building NumNet    ##
echo                   ##########################
echo[
echo[

dotnet build -c Release

echo NumNet was built in release configuration, ready to create package
pause
echo[
echo[
echo                 #################################
echo                 ##   Building NumNet Package   ##
echo                 #################################
echo[
echo[

nuget pack Proxem.NumNet/nuspec/Proxem.NumNet.nuspec -Symbols -OutputDirectory Proxem.NumNet/nuspec/

echo Package was created in folder Proxem.NumNet/nuspec/
pause