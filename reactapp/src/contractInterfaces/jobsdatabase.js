import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x839135A064717f9c430ebb9f0382F8305c317DDF';

export default new web3.eth.Contract(jobs, address);
